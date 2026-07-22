import numpy as np
import wandb
import torch
import traceback

from matplotlib import pyplot as plt
from preprocess_database import inverse_transform_signals
from utils import add_combined_noise

# Clinical metrics imports
from metrics.DF import calculate_df_batch
from functions.NLEO_functions import calculateNLEORaw
from functions.signalProcessingFunctions import (
    calculateUnipolarEGMSlopeParallel, detectLATs, LATSettingsClass
)
from itertools import combinations
from utils import calculate_lat_matching


def plot_reconstruction(sample_data, reconstructed_data, category="reconstructions"):
    """
    Registrar señales originales y reconstruidas en W&B bajo categorías específicas.

    Args:
        sample_data (array): Datos originales.
        reconstructed_data (array): Datos reconstruidos por el modelo.
        category (str): Categoría principal en wandb (default 'reconstructions').
        log_to_wandb (bool): Si es True, registra las gráficas en W&B.
    """
    for i, (original, reconstructed) in enumerate(zip(sample_data, reconstructed_data)):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(original.squeeze(), label="Original", linestyle="-")
        plt.plot(reconstructed.squeeze(), label="Reconstructed", linestyle="-", alpha=0.7)
        plt.title(f"Sample {i + 1}: Original vs Reconstructed")
        plt.xlabel("Time Points")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid()

        # Guardar en wandb solo si está activo
        if wandb.run:
            wandb.log({f"{category}/Sample_{i + 1}": wandb.Image(fig)})

        plt.close()



def plot_multiple_reconstructions_with_latent(original, reconstructed, latent_vectors, config,
                         num_samples=3, category="reconstructions", alpha=0.7, num_iterations=3):

    print(f"DEBUG plot_multiple: original shape={original.shape}, reconstructed shape={reconstructed.shape}, latent shape={latent_vectors.shape}, num_samples={num_samples}")

    total_samples = original.shape[0]
    offset = 2 if config.model_type == 'UNIPOLAR' else 0.5
    offset_latent = 1 # Offset vertical entre latentes

    rng = np.random.default_rng(seed=42)
    for iteration in range(num_iterations):
        selected_indices = rng.choice(total_samples, num_samples, replace=False)

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f"{config.model_type}", fontsize=16)
        colors = plt.cm.viridis(np.linspace(0, 1, num_samples))

        # Graficar señales originales vs reconstruidas con offset
        for i, idx in enumerate(selected_indices):
            axes[0].plot(original[idx].squeeze() + i * offset, linestyle="--", color="black", alpha=alpha,
                         label=f"Original {i + 1}" if i == 0 else "")
            axes[0].plot(reconstructed[idx].squeeze() + i * offset, linestyle="-", color=colors[i], alpha=alpha,
                         label=f"Reconstructed {i + 1}")

        axes[0].set_title("Original vs Reconstructed Signals")
        axes[0].set_xlabel("Time Points")
        axes[0].set_ylabel("Amplitude (offset applied)")
        axes[0].grid()

        # Graficar valores en el espacio latente con offset
        num_latent_dims = latent_vectors.shape[1]
        x_positions = np.arange(num_latent_dims)

        #print(f"DEBUG iteration {iteration}: selected_indices={selected_indices}, num_latent_dims={num_latent_dims}")

        # Graficar las barras con offset real (usando los mismos índices seleccionados)
        for i, idx in enumerate(selected_indices):
            base_offset = i * offset_latent  # Desplazamiento de la base
            latent_vals = latent_vectors[idx]
            #print(f"DEBUG: i={i}, idx={idx}, latent_vals range=[{latent_vals.min():.4f}, {latent_vals.max():.4f}], base_offset={base_offset}")
            axes[1].bar(x_positions, latent_vals, bottom=base_offset, width=0.6, alpha=0.8,
                   color=colors[i], label=f"Latent {i + 1}")

        # Ajustes del gráfico
        axes[1].set_title("Latent Space Representation (normalized)")
        axes[1].set_xlabel("Latent Dimension")
        axes[1].set_ylabel("Activation Value (offset applied)")
        axes[1].set_xticks(np.arange(0, num_latent_dims, 4))
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)
        axes[1].legend(loc='upper right')

        if wandb.run:
            wandb_key = f"{category}/reconstructed_signals_{iteration + 1}"
            wandb.log({wandb_key: wandb.Image(fig)})

        plt.close()

def plot_original_and_noise(
    original: np.ndarray,
    reconstructed: np.ndarray,
    noisy: np.ndarray,
    *,
    num_samples: int = 10,
    category: str = "test",
    subfolder: str = "orig_noisy_rec",
    noise_info: dict = None
) -> None:
    """
    Plot original, noisy, and reconstructed signals.

    Args:
        original: Original clean signals
        reconstructed: Reconstructed signals from the model
        noisy: Noisy input signals
        num_samples: Number of samples to plot
        category: Category for wandb logging
        subfolder: Subfolder for wandb logging
        noise_info: Dict with 'types' (list of noise type names) and 'snr_db' (float)
    """
    # --- comprobaciones básicas ---------------------------------------------
    if any(arr is None for arr in (original, reconstructed, noisy)):
        raise ValueError("Debes proporcionar 'original', 'reconstructed' y 'noisy'.")
    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)
    noisy = np.asarray(noisy)

    # --- elección de muestras aleatorias -------------------------------------
    n = min(num_samples, original.shape[0])
    idxs = np.random.default_rng(seed=42).choice(original.shape[0], n, replace=False)

    # Build noise info string for title
    noise_str = ""
    if noise_info is not None:
        noise_types_str = ", ".join(noise_info.get('types', []))
        snr_db = noise_info.get('snr_db')
        if snr_db is not None:
            noise_str = f"\nNoise: [{noise_types_str}] | SNR: {snr_db:.1f} dB"
        else:
            noise_str = f"\nNoise: [{noise_types_str}]"

    for k, idx in enumerate(idxs):
        fig = plt.figure(figsize=(10, 4))

        plt.plot(
            original[idx].squeeze(),
            label="Original",
            lw=1.5,
            color="black",
            zorder=2
        )

        # Noisy: rojo muy translúcido y fino, en el fondo
        plt.plot(
            noisy[idx].squeeze(),
            label="Noisy",
            lw=0.6,  # línea fina
            color="red",
            alpha=0.33,  # casi transparente
            zorder=1  # se dibuja al fondo
        )

        plt.plot(
            reconstructed[idx].squeeze(),
            label="Reconstructed",
            lw=1.5,
            color="orange",
            alpha=0.8,
            zorder=3
        )

        title = f"Sample {idx + 1} — Original vs Noisy vs Reconstructed{noise_str}"
        plt.title(title)
        plt.xlabel("Time Points")
        plt.ylabel("Electrogram voltage (mV)")
        plt.legend()
        plt.grid(True)

        # Registro en Weights & Biases (si la sesión está activa)
        if wandb.run is not None:
            wandb.log({f"{category}/{subfolder}/sample_{idx + 1}": wandb.Image(fig)})

        plt.close(fig)



def visualize_reconstructions(model, data, p_inf, p_sup, label, num_samples, args, device='cuda'):
    """Visualize reconstructions for a set of samples"""
    print(f"\nGenerating {label} reconstruction visualizations...")

    indices = np.random.default_rng(seed=42).choice(len(data), num_samples, replace=False)
    sample_data = data[indices].to(torch.float32).to(device)

    with torch.no_grad():
        reconstructed = model(sample_data).cpu().numpy()

        # For autoencoders, also get latent vectors
        if hasattr(model, 'encoder'):
            encoder_output = model.encoder(sample_data)
            # Handle CLARAE_V3, V4 which returns (latent, skips)
            if isinstance(encoder_output, tuple):
                latent_vecs = encoder_output[0].cpu().numpy()
                #print(f"DEBUG: CLARAE_V3, V4 detected - latent shape: {latent_vecs.shape}")
            else:
                latent_vecs = encoder_output.cpu().numpy()
                #print(f"DEBUG: Regular encoder - latent shape: {latent_vecs.shape}")
        else:
            latent_vecs = None
            print(f"DEBUG: No encoder found")

    # Inverse transform to original scale
    orig_inv = inverse_transform_signals(sample_data.cpu().numpy(), p_inf, p_sup)
    rec_inv = inverse_transform_signals(reconstructed.squeeze(1), p_inf, p_sup)

    # Plot reconstructions
    plot_reconstruction(orig_inv, rec_inv, label)

    # Plot multiple reconstructions with latent space if available
    if latent_vecs is not None:
        # Create a simple config object for plotting
        class PlotConfig:
            def __init__(self, model_type):
                self.model_type = model_type

        plot_config = PlotConfig('UNIPOLAR' if args.unipolar else 'BIPOLAR')
        plot_multiple_reconstructions_with_latent(
            orig_inv, rec_inv, latent_vecs, plot_config,
            num_samples=3, category=label
        )

    print(f"{label.capitalize()} visualizations completed.")


def visualize_test_with_noise_info(model, data, p_inf, p_sup, label, num_samples, args, device='cuda', is_bipolar=False):
    """
    Visualize test set reconstructions with 4 subplots:
    - Row 1: Original vs Reconstructed (clean input) + Vpp
    - Row 2: Original vs Noisy vs Reconstructed with noise info (type, SNR) + Vpp
    - Row 3: DF Spectrum comparison (original vs reconstructed)
    - Row 4: NLEO comparison (bipolar) or LAT comparison (unipolar)

    This function generates samples with DIFFERENT noise combinations for each sample,
    ensuring variety in the visualization (gaussian, baseline_wander, powerline, spike, and combinations).

    Args:
        model: Trained model
        data: Test data tensor
        p_inf: Lower percentile for inverse normalization
        p_sup: Upper percentile for inverse normalization
        label: Label for the visualizations (e.g., "test")
        num_samples: Number of samples to visualize
        args: Arguments containing noise parameters
        device: Device to use for computation
        is_bipolar: Whether signals are bipolar (affects row 4: NLEO vs LAT)
    """

    print(f"\nGenerating {label} visualizations with noise info subplot...")
    print(f"  Noise SNR range: [{args.noise_snr_min:.1f}, {args.noise_snr_max:.1f}] dB")
    print(f"  Signal type: {'BIPOLAR' if is_bipolar else 'UNIPOLAR'}")

    # Get noise configuration from args
    fs = getattr(args, 'sampling_freq', 500)
    powerline_freq = getattr(args, 'powerline_freq', 50)

    # Generate all possible noise combinations (1 type, 2 types, etc.)
    all_noise_types = ['gaussian', 'baseline_wander', 'powerline', 'spike']
    noise_combinations = []

    # Single noise types first (4 combinations)
    for noise_type in all_noise_types:
        noise_combinations.append([noise_type])

    # Two noise types (6 combinations)
    for combo in combinations(all_noise_types, 2):
        noise_combinations.append(list(combo))

    # Three noise types (4 combinations)
    for combo in combinations(all_noise_types, 3):
        noise_combinations.append(list(combo))

    # All four noise types (1 combination)
    noise_combinations.append(all_noise_types.copy())

    # Total: 15 combinations, cycle through them
    print(f"  Will cycle through {len(noise_combinations)} noise combinations")

    # Select fixed samples (seed=42 ensures the same signals are always used,
    # enabling consistent comparison across runs/models)
    rng = np.random.default_rng(seed=42)
    np.random.seed(42)
    indices = rng.choice(len(data), num_samples, replace=False)

    model.eval()

    # Process each sample with a different noise combination
    for k, idx in enumerate(indices):
        # Get single sample
        sample = data[idx:idx+1].to(torch.float32).to(device)

        # Select noise combination for this sample (cycle through combinations)
        noise_combo = noise_combinations[k % len(noise_combinations)]

        # Add noise to sample with this specific combination
        noisy_sample, noise_info = add_combined_noise(
            sample,
            noise_types=noise_combo,
            snr_db_min=args.noise_snr_min,
            snr_db_max=args.noise_snr_max,
            fs=fs,
            powerline_freq=powerline_freq,
            return_info=True
        )

        print(f"  Sample {k+1}: Noise=[{', '.join(noise_info['types'])}], SNR={noise_info['snr_db']:.1f} dB")

        # Generate reconstructions
        with torch.no_grad():
            reconstructed_clean = model(sample).cpu().numpy()
            reconstructed_noisy = model(noisy_sample).cpu().numpy()

        # Inverse transform to original scale
        original_inv = inverse_transform_signals(sample.cpu().numpy(), p_inf, p_sup)
        noisy_inv = inverse_transform_signals(noisy_sample.cpu().numpy(), p_inf, p_sup)
        reconstructed_clean_inv = inverse_transform_signals(reconstructed_clean.squeeze(1), p_inf, p_sup)
        reconstructed_noisy_inv = inverse_transform_signals(reconstructed_noisy.squeeze(1), p_inf, p_sup)

        # Build noise info string
        noise_types_str = ", ".join(noise_info.get('types', []))
        snr_db = noise_info.get('snr_db', 0)

        # Extract numpy arrays for metrics calculation
        original_np = original_inv[0].squeeze()
        recon_clean_np = reconstructed_clean_inv[0].squeeze()
        recon_noisy_np = reconstructed_noisy_inv[0].squeeze()

        # Time axis in seconds
        t = np.arange(len(original_np)) / fs

        # Calculate Vpp
        vpp_orig = np.max(original_np) - np.min(original_np)
        vpp_clean = np.max(recon_clean_np) - np.min(recon_clean_np)
        vpp_noisy = np.max(recon_noisy_np) - np.min(recon_noisy_np)

        # Calculate R²
        ss_tot = np.sum((original_np - np.mean(original_np)) ** 2)
        ss_res_clean = np.sum((original_np - recon_clean_np) ** 2)
        ss_res_noisy = np.sum((original_np - recon_noisy_np) ** 2)
        r2_clean = 1.0 - ss_res_clean / ss_tot if ss_tot > 0 else 0.0
        r2_noisy = 1.0 - ss_res_noisy / ss_tot if ss_tot > 0 else 0.0

        # Calculate DF
        try:
            df_result_orig = calculate_df_batch(original_np.reshape(1, -1), fs=fs, verbose=False)
            df_result_noisy = calculate_df_batch(recon_noisy_np.reshape(1, -1), fs=fs, verbose=False)
            df_orig = df_result_orig.DF_values[0]
            df_recon = df_result_noisy.DF_values[0]
            df_error = abs(df_orig - df_recon)
            df_valid = True
        except Exception as e:
            print(f"    Warning: DF calculation failed: {e}")
            df_valid = False

        # ── publication style helper ─────────────────────────────────────────
        def _style(ax):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.8)
            ax.spines['bottom'].set_linewidth(0.8)
            ax.set_facecolor('#f9f9f9')
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4, color='#aaaaaa', zorder=0)
            ax.tick_params(labelsize=16, width=0.8, direction='out')

        FS_TITLE  = 19   # subplot title
        FS_LABEL  = 17   # axis labels
        FS_LEGEND = 16   # legend
        FS_ANNOT  = 16   # annotation text

        # Create figure with 4 subplots
        fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=False)
        fig.patch.set_facecolor('white')

        # Row 1: Original vs Reconstructed (clean) + Vpp
        _style(axes[0])
        axes[0].plot(t, original_np, label="Original", lw=2.5, color="black", alpha=0.88)
        axes[0].plot(t, recon_clean_np, label="Reconstructed (clean)",
                     lw=2.3, color="#0057B8", alpha=0.9)
        axes[0].set_title(f"Sample {k + 1} – Clean Reconstruction", fontsize=FS_TITLE, fontweight='bold')
        axes[0].set_xlabel("Time (s)", fontsize=FS_LABEL)
        axes[0].set_ylabel("mV", fontsize=FS_LABEL)
        axes[0].legend(loc='upper left', fontsize=FS_LEGEND, framealpha=0.35, edgecolor='#cccccc')
        axes[0].text(0.985, 0.97, f'Vpp: {vpp_orig:.2f} / {vpp_clean:.2f} mV',
                     transform=axes[0].transAxes, fontsize=FS_ANNOT, ha='right', va='top',
                     fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#dddddd', alpha=0.9))
        axes[0].text(0.985, 0.84, f'R²: {r2_clean:.4f}',
                     transform=axes[0].transAxes, fontsize=FS_ANNOT, ha='right', va='top',
                     fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#dddddd', alpha=0.9))

        # Row 2: Original vs Noisy vs Reconstructed (denoised) + Vpp
        _style(axes[1])
        axes[1].plot(t, noisy_inv[0].squeeze(), label="Noisy input", lw=1.2, color="#FF0000",
                     alpha=0.55, zorder=1)
        axes[1].plot(t, original_np, label="Original", lw=2.5, color="black", alpha=0.88, zorder=2)
        axes[1].plot(t, recon_noisy_np, label="Reconstructed (denoised)",
                     lw=2.3, color="#E85D00", alpha=0.9, zorder=3)
        axes[1].set_title(f"Denoising – Noise: [{noise_types_str}]  |  SNR: {snr_db:.1f} dB",
                          fontsize=FS_TITLE, fontweight='bold')
        axes[1].set_xlabel("Time (s)", fontsize=FS_LABEL)
        axes[1].set_ylabel("mV", fontsize=FS_LABEL)
        axes[1].legend(loc='upper left', fontsize=FS_LEGEND, framealpha=0.35, edgecolor='#cccccc')
        axes[1].text(0.985, 0.97, f'Vpp: {vpp_orig:.2f} / {vpp_noisy:.2f} mV',
                     transform=axes[1].transAxes, fontsize=FS_ANNOT, ha='right', va='top',
                     fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#dddddd', alpha=0.9))
        axes[1].text(0.985, 0.84, f'R²: {r2_noisy:.4f}',
                     transform=axes[1].transAxes, fontsize=FS_ANNOT, ha='right', va='top',
                     fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#dddddd', alpha=0.9))

        # Row 3: DF Spectrum comparison
        _style(axes[2])
        if df_valid:
            freq_axis = np.arange(len(df_result_orig.DF_Spectrum[0])) * fs / df_result_orig.DF_Nfft[0]
            axes[2].plot(freq_axis, df_result_orig.DF_Spectrum[0], color='black',
                        label=f'Original (DF={df_orig:.1f} Hz)', lw=2.3, alpha=0.88)
            axes[2].plot(freq_axis, df_result_noisy.DF_Spectrum[0], color='#E85D00',
                        label=f'Reconstructed (DF={df_recon:.1f} Hz)', lw=2.1, alpha=0.85)
            axes[2].axvline(x=df_orig,  color='black',   linestyle='--', lw=1.0, alpha=0.6)
            axes[2].axvline(x=df_recon, color='#E85D00', linestyle='--', lw=1.0, alpha=0.6)
            axes[2].set_xlim([0, 20])
            axes[2].set_xlabel('Frequency (Hz)', fontsize=FS_LABEL)
            axes[2].set_ylabel('Power Spectral Density (PSD)', fontsize=FS_LABEL)
            axes[2].set_title(f'Dominant Frequency Comparison  |  DF Error: {df_error:.2f} Hz',
                              fontsize=FS_TITLE, fontweight='bold')
            axes[2].legend(loc='upper left', fontsize=FS_LEGEND, framealpha=0.35, edgecolor='#cccccc')
        else:
            axes[2].text(0.5, 0.5, 'DF calculation failed', transform=axes[2].transAxes,
                        ha='center', va='center', fontsize=FS_TITLE)
            axes[2].set_title('Dominant Frequency Comparison', fontsize=FS_TITLE, fontweight='bold')

        # Row 4: NLEO (bipolar) or LAT (unipolar)
        _style(axes[3])
        if is_bipolar:
            try:
                nleo_orig = calculateNLEORaw(original_np.reshape(1, -1), fs=fs)
                nleo_recon = calculateNLEORaw(recon_noisy_np.reshape(1, -1), fs=fs)
                nleo_corr = np.corrcoef(nleo_orig[0], nleo_recon[0])[0, 1]

                t_nleo = np.arange(len(nleo_orig[0])) / fs
                axes[3].plot(t_nleo, nleo_orig[0],  color='black',   lw=2.3, alpha=0.88, label='Original NLEO')
                axes[3].plot(t_nleo, nleo_recon[0], color='#E85D00', lw=2.1, alpha=0.85, label='Reconstructed NLEO')

                axes[3].set_xlabel('Time (s)', fontsize=FS_LABEL)
                axes[3].set_ylabel('NLEO', fontsize=FS_LABEL)
                axes[3].set_title(f'NLEO Raw Comparison (Bipolar)  |  Correlation: {nleo_corr:.4f}',
                                  fontsize=FS_TITLE, fontweight='bold')
                axes[3].legend(loc='upper left', fontsize=FS_LEGEND, framealpha=0.35, edgecolor='#cccccc')
            except Exception as e:
                print(f"    Warning: NLEO calculation failed: {e}")
                axes[3].text(0.5, 0.5, 'NLEO calculation failed', transform=axes[3].transAxes,
                            ha='center', va='center', fontsize=FS_TITLE)
                axes[3].set_title('NLEO Comparison (Bipolar)', fontsize=FS_TITLE, fontweight='bold')
        else:
            try:
                LAT_M            = 10
                LAT_TAU          = 0.00035
                LAT_BLANK_PERIOD = 0.045
                LAT_SIGMA_ABS_TH = 0.05
                LAT_TOLERANCE    = 5

                lat_settings = LATSettingsClass(
                    M=LAT_M, fs=fs,
                    tau_input=LAT_TAU,
                    blank_period_time=LAT_BLANK_PERIOD,
                    sigma_abs_th=LAT_SIGMA_ABS_TH,
                )

                slopes_orig  = calculateUnipolarEGMSlopeParallel(original_np.reshape(1, -1), LAT_M)
                slopes_recon = calculateUnipolarEGMSlopeParallel(recon_noisy_np.reshape(1, -1), LAT_M)

                beta_orig = -1 * slopes_orig[0];  beta_orig[beta_orig < 0] = 0
                beta_recon = -1 * slopes_recon[0]; beta_recon[beta_recon < 0] = 0

                lat_det_orig  = detectLATs(beta_orig,  lat_settings)
                lat_det_recon = detectLATs(beta_recon, lat_settings)

                lats_orig_idx  = np.array([int(x) for x in lat_det_orig.activation_peaks_indices],  dtype=np.intp)
                lats_recon_idx = np.array([int(x) for x in lat_det_recon.activation_peaks_indices], dtype=np.intp)
                lats_orig_pos  = lats_orig_idx / fs
                lats_recon_pos = lats_recon_idx / fs
                lats_orig  = lats_orig_idx
                lats_recon = lats_recon_idx

                lat_result      = calculate_lat_matching(lats_orig, lats_recon, tolerance=LAT_TOLERANCE)
                n_matched       = len(lat_result['matched_diffs'])
                mae_ms          = lat_result['matched_mae_ms']
                n_unmatched_orig = lat_result['n_unmatched_orig']

                time_pts = np.arange(len(beta_orig)) / fs
                axes[3].plot(time_pts, beta_orig,  lw=2.3, color='black',   alpha=0.88, label='β+[n] Orig')
                axes[3].plot(time_pts, beta_recon, lw=2.1, color='#E85D00', alpha=0.80, label='β+[n] Recon')

                y_min = min(0, beta_orig.min(), beta_recon.min())

                if len(lats_orig) > 0:
                    axes[3].vlines(lats_orig_pos, ymin=y_min, ymax=beta_orig[lats_orig],
                                   colors='green', linewidth=1.5, alpha=0.2, label='LAT Orig')
                    axes[3].scatter(lats_orig_pos, beta_orig[lats_orig],
                                    color='green', s=60, marker='*', zorder=5)
                if len(lats_recon) > 0:
                    axes[3].vlines(lats_recon_pos, ymin=y_min, ymax=beta_recon[lats_recon],
                                   colors='red', linewidth=1.5, alpha=0.2, label='LAT Recon')
                    axes[3].scatter(lats_recon_pos, beta_recon[lats_recon],
                                    color='red', s=60, marker='*', zorder=5)

                axes[3].set_xlabel('Time (s)', fontsize=FS_LABEL)
                axes[3].set_ylabel('V/ms', fontsize=FS_LABEL)
                axes[3].set_title(
                    f'LAT Detection  |  Orig: {len(lats_orig)}, Recon: {len(lats_recon)}  |  '
                    f'Matched: {n_matched} (MAE={mae_ms:.1f} ms)  |  Unmatched orig: {n_unmatched_orig}',
                    fontsize=FS_TITLE, fontweight='bold'
                )
                axes[3].legend(loc='upper left', fontsize=FS_LEGEND, framealpha=0.35, edgecolor='#cccccc')
                y_min_axis = min(0, beta_orig.min(), beta_recon.min())
                axes[3].set_ylim(bottom=y_min_axis, top=max(beta_orig.max(), beta_recon.max()) * 1.05)
            except Exception as e:
                print(f"    Warning: LAT calculation failed: {e}")
                traceback.print_exc()
                axes[3].text(0.5, 0.5, 'LAT calculation failed', transform=axes[3].transAxes,
                            ha='center', va='center', fontsize=FS_TITLE)
                axes[3].set_title('LAT Detection (Unipolar)', fontsize=FS_TITLE, fontweight='bold')

        plt.tight_layout()

        # Log to wandb
        if wandb.run is not None:
            wandb.log({f"{label}/reconstruction_with_noise/sample_{k + 1}": wandb.Image(fig)})

        plt.close(fig)

    print(f"{label.capitalize()} visualizations with noise info completed.")