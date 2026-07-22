"""
Non-Linear Energy Operator (NLEO) Metric for EGM Signals
=========================================================

This module calculates NLEO for EGM signals to evaluate activation pattern
preservation. Can be used to assess reconstruction_denoising quality by comparing
NLEO between original and reconstructed signals.

NLEO[n] = x[n]^2 - x[n-1] * x[n+1]

NLEO enhances rapid signal transitions and is useful for evaluating
if activation morphology is preserved after reconstruction_denoising.

Metric used: Pearson correlation — higher is better (↑).
  corr = Pearson(NLEO_orig, NLEO_recon)  per signal
  Computed on original-scale (denormalized) signals.
"""

import numpy as np
import matplotlib.pyplot as plt

from functions.NLEO_functions import calculateNLEO, calculateNLEORaw
from utils import load_patient_signals  # Unified data loading function


# ==============================================================================
# NLEO Calculation Functions
# ==============================================================================


def calculate_nleo_corr_batch(original: np.ndarray, reconstructed: np.ndarray,
                              batch_size: int = 5000) -> dict:
    """
    Calculate Pearson correlation of NLEO Raw between original and reconstructed signals.

    corr = Pearson(NLEO_orig, NLEO_recon)  per signal.
    Higher is better (↑). Should be called on original-scale (denormalized) signals.

    Processes in mini-batches to limit memory use.

    Args:
        original: 2D array (N, L) of original signals (denormalized, original scale)
        reconstructed: 2D array (N, L) of reconstructed signals (denormalized)
        batch_size: Signals per mini-batch (default: 5000)

    Returns:
        dict with 'mean' and 'std' of per-signal Pearson correlation values
    """
    from tqdm import tqdm

    n_signals = original.shape[0]
    corrs = np.empty(n_signals, dtype=np.float64)

    for start in tqdm(range(0, n_signals, batch_size), desc="  NLEO corr", unit="batch"):
        end = min(start + batch_size, n_signals)
        nleo_o = calculateNLEORaw(original[start:end])
        nleo_r = calculateNLEORaw(reconstructed[start:end])
        for i in range(end - start):
            c = np.corrcoef(nleo_o[i], nleo_r[i])[0, 1]
            corrs[start + i] = 0.0 if np.isnan(c) else float(c)
        del nleo_o, nleo_r

    return {'mean': float(np.mean(corrs)), 'std': float(np.std(corrs))}


def get_nleo_statistics(nleo_signals: np.ndarray) -> dict:
    """
    Compute descriptive statistics for NLEO values.

    Args:
        nleo_signals: 2D array of NLEO signals

    Returns:
        Dictionary with keys: mean, std, min, max, median
    """
    return {
        'mean': float(np.mean(nleo_signals)),
        'std': float(np.std(nleo_signals)),
        'min': float(np.min(nleo_signals)),
        'max': float(np.max(nleo_signals)),
        'median': float(np.median(nleo_signals))
    }

# ==============================================================================
# Plotting Functions
# ==============================================================================

def plot_signal_with_nleo(signal: np.ndarray, nleo: np.ndarray,
                          nleo_binary: np.ndarray, nleo_activity: float,
                          fs: int = 500, title: str = None, save_path: str = None):
    """
    Plot a signal with NLEO Raw and Binary Activity overlay.

    Args:
        signal: 1D numpy array representing the EGM signal
        nleo: Raw NLEO signal
        nleo_binary: Binary activation map
        nleo_activity: Activity ratio (scalar)
        fs: Sampling frequency in Hz (default: 500)
        title: Optional title for the plot
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Time axis in ms
    time_ms = np.arange(len(signal)) / fs * 1000

    # --- Plot 1: Original signal ---
    ax1 = axes[0]
    ax1.plot(time_ms, signal, 'b-', linewidth=0.8)
    ax1.set_ylabel('Amplitude (mV)')
    ax1.set_title(f'EGM Signal (Bipolar) + Binary Activity (Activity ratio: {nleo_activity:.4f})')
    nleo_max = np.max(signal) if np.max(signal) > 0 else 1.0
    nleo_min = np.min(signal) if np.min(signal) < 0 else -1.0
    ax1.fill_between(time_ms, nleo_binary * nleo_max, color='orange', alpha=0.4, label='Binary Activity')
    ax1.fill_between(time_ms, nleo_binary * nleo_min, color='orange', alpha=0.4, label='Binary Activity')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: NLEO Raw with Binary Activity overlay ---
    ax2 = axes[1]
    ax2.plot(time_ms, nleo, 'g-', linewidth=0.8, label='NLEO Raw')
    # Binary Activity as fill_between scaled to NLEO max
    ax2.set_ylabel('NLEO')
    ax2.set_xlabel('Time (ms)')
    ax2.set_title('NLEO Raw')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Main title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to: {save_path}")


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser(
        description='Calculate and plot NLEO for EGM signals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -m metrics.NLEO --patient_file "DATABASE/EGMS_xz/Patient 2021_09_30_EGMs.xz" --map_index 0
        """
    )
    parser.add_argument('--patient_file', type=str,
                        default='DATABASE/EGMS_xz/Patient 2021_09_30_EGMs.xz',
                        help='Path to .xz patient file')
    parser.add_argument('--map_index', type=int, default=0,
                        help='Index of map to load (default: 0)')
    parser.add_argument('--output_dir', type=str, default='results_plots/NLEO',
                        help='Output directory for plots (default: results_plots/NLEO)')
    parser.add_argument('--num_plots', type=int, default=3,
                        help='Number of random signals to plot (default: 3)')
    parser.add_argument('--fs', type=int, default=500,
                        help='Sampling frequency in Hz (default: 500)')

    args = parser.parse_args()

    print("NLEO Metric - Non-Linear Energy Operator Analysis")
    print("=" * 55)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load bipolar signals
    print(f"\nLoading signals from: {args.patient_file}")
    signals_bip, patient_id, map_name = load_patient_signals(
        args.patient_file, map_index=args.map_index, signal_type='bipolar'
    )
    print(f"  Patient: {patient_id}")
    print(f"  Map: {map_name}")
    print(f"  Bipolar signals shape: {signals_bip.shape}")

    # Calculate NLEO for bipolar (full pipeline: raw, LPF, binary activity, activity ratio)
    print("\nCalculating NLEO (Bipolar)...")
    nleo_bip, nleo_lpf_bip, nleo_binary_bip, nleo_activity_bip = calculateNLEO(signals_bip, NLEO_threshold = 0.5, Merge_distance= 5)

    # Print statistics for bipolar
    print(f"\n{'='*55}")
    print("BIPOLAR SIGNALS")
    print(f"{'='*55}")
    print(f"\nNLEO Raw Statistics (n={signals_bip.shape[0]} signals):")
    print("-" * 40)
    for key, value in get_nleo_statistics(nleo_bip).items():
        print(f"  {key}: {value:.6f}")
    print(f"\nNLEO Activity Ratio (mean): {np.mean(nleo_activity_bip):.4f}")
    print(f"NLEO Activity Ratio (std):  {np.std(nleo_activity_bip):.4f}")

    # Select random indices for plotting
    num_signals = signals_bip.shape[0]
    random_indices = np.random.choice(num_signals, size=min(args.num_plots, num_signals), replace=False)

    # Clean map name for filename
    clean_map_name = map_name.replace('/', '_').replace(' ', '_')

    # Plot random bipolar signals
    print(f"\nPlotting {len(random_indices)} random bipolar signals...")
    for idx in random_indices:
        plot_signal_with_nleo(
            signal=signals_bip[idx],
            nleo=nleo_bip[idx],
            nleo_binary=nleo_binary_bip[idx],
            nleo_activity=float(nleo_activity_bip[idx]),
            fs=args.fs,
            title=f'{patient_id} - {map_name} - Bipolar #{idx}',
            save_path=f'{args.output_dir}/{patient_id}_{clean_map_name}_bipolar_{idx}_NLEO.png'
        )

    print("\nDone!")
