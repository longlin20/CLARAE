"""
run_test.py
===========
Evaluate one or more pre-trained autoencoder models on the test split.
Architecture and hyperparameters are resolved automatically from the .pth filename
via MODEL_REGISTRY. Fixed hyperparameters match run_experiments.py COMMON.

Usage
-----
# Single model (architecture and loss resolved from filename via MODEL_REGISTRY)
python run_test.py \
    --pth_path model_pth/UNIPOLAR_CLARAE_SCM_GLU.pth \
    --preprocessed_data_dir processed_data_final/

# Multiple models
python run_test.py \
    --pth_path model_pth/UNIPOLAR_CLARAE_SCM_GLU.pth model_pth/BIPOLAR_ACDAE.pth \
    --preprocessed_data_dir processed_data_final/

# Override loss function for a specific run
python run_test.py \
    --pth_path model_pth/UNIPOLAR_CLARAE_SCM_GLU.pth \
    --preprocessed_data_dir processed_data_final/ \
    --loss_function dtw
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import wandb
from glob import glob
from eval_utils import load_split_h5

from training_engine import validate
from plots.plot_AE_results import visualize_test_with_noise_info
from utils import add_random_noise, add_combined_noise

# Clinical metrics
from metrics.R2_MSE import calculate_mse_batch
from metrics.Vpp import calculate_vpp_error_batch
from metrics.DF import calculate_df_error_batch
from metrics.NLEO import calculate_nleo_corr_batch
from metrics.LATs_unipolar import calculate_lat_metrics_batch
from tqdm import tqdm

from run_training import compile_model, set_seed


# =============================================================================
# FIXED HYPERPARAMETERS  (mirror of run_experiments.py COMMON)
# Architecture, signal type, and loss_function are resolved from MODEL_REGISTRY.
# =============================================================================

FIXED_PARAMS = {
    # Model architecture hyperparameters
    "latent_dim":      64,
    "filters_initial": 64,
    "dropout_rate":    0.1,
    "skip_dropout":    0.7,
    "gate_init":       -5.0,
    "dense_dim":       128,
    "q_parameter":     2,
    # Loss hyperparameters (DTW / DTW+Slope)
    "dtw_gamma":       1.0,
    "slope_alpha":     0.5,
    "slope_m":         5,
    # Noise evaluation
    "random_noise":    True,
    "noise_types":     ["gaussian", "baseline_wander", "powerline", "spike"],
    "min_noise_types": 1,
    "max_noise_types": 4,
    "noise_snr_min":   -5.0,
    "noise_snr_max":   10.0,
    "powerline_freq":  50,
    "sampling_freq":   500,
    # Reproducibility
    "seed":            42,
    "noise_seed":      0,
    # Inference
    "batch_size":      256,
    # Dummy fields required by compile_model (not used during test)
    "optimizer":       "Adam",
    "learning_rate":   0.0002,
    "weight_decay":    1e-3,
    "add_noise":       False,
}

# =============================================================================
# MODEL REGISTRY
# Maps the architecture key (pth stem with UNIPOLAR_/BIPOLAR_ prefix stripped)
# to its model_architecture string and optional hyperparameter overrides.
#
# pth naming convention from run_experiments.py: {UNIPOLAR|BIPOLAR}_{arch_key}.pth
# Add per-model overrides as extra keys if a model was trained with different params.
# =============================================================================

from model_registry import (
    MODEL_REGISTRY_TEST as MODEL_REGISTRY,
    _KNOWN_LOSSES, _KNOWN_NOISE_TAGS, _strip_loss_noise_suffix,
)


# ---------------------------------------------------------------------------
# Clinical metric helpers
# ---------------------------------------------------------------------------


def _denormalize(signals, p_inf, p_sup):
    """Inverse of: normalized = 2*(x - p_inf)/(p_sup - p_inf) - 1  →  range [-1, 1]."""
    return ((signals + 1) / 2) * (p_sup - p_inf) + p_inf


def calculate_clinical_metrics(model, test_data, device, fs=500, is_bipolar=False, batch_size=256,
                               add_noise=False, noise_config=None, metrics=None, percentiles=None):
    """
    Calculate clinical metrics comparing original vs reconstructed signals.

    Args:
        model: The trained model
        test_data: Clean test data tensor (normalized [-1, 1])
        device: Device to run inference on
        fs: Sampling frequency
        is_bipolar: Whether signals are bipolar (affects which metrics are calculated)
        batch_size: Batch size for inference
        add_noise: If True, add noise to input before passing to model
        noise_config: Dict with noise parameters (snr_min, snr_max, noise_types, etc.)
        metrics: Set/list of metric names to compute. None = all.
                 Choices: 'mse', 'vpp', 'df', 'lat' (unipolar), 'nleo' (bipolar).
        percentiles: [p_inf, p_sup] used for normalization. When provided, Vpp / LAT /
                     NLEO are computed on denormalized (original-scale) signals.

    Returns dict with mean and std for each metric:
    - MSE, Vpp error, DF error, NLEO NRMSE (only for bipolar), LAT (only for unipolar)
    """
    if metrics is not None:
        metrics = set(m.lower() for m in metrics)
    model.eval()

    # Get original data (always clean for comparison)
    original = test_data.cpu().numpy()
    reconstructed_list = []

    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size].to(device)

            # Optionally add noise to input
            if add_noise and noise_config:
                if noise_config.get('random_noise', False):
                    batch = add_random_noise(
                        batch,
                        snr_db_min=noise_config['snr_min'],
                        snr_db_max=noise_config['snr_max'],
                        min_types=noise_config.get('min_types', 1),
                        max_types=noise_config.get('max_types', 4),
                        fs=fs,
                        powerline_freq=noise_config.get('powerline_freq', 50)
                    )
                else:
                    batch = add_combined_noise(
                        batch,
                        noise_types=noise_config.get('noise_types', ['gaussian']),
                        snr_db_min=noise_config['snr_min'],
                        snr_db_max=noise_config['snr_max'],
                        fs=fs,
                        powerline_freq=noise_config.get('powerline_freq', 50)
                    )

            recon_batch = model(batch).cpu().numpy()
            reconstructed_list.append(recon_batch)

    reconstructed = np.concatenate(reconstructed_list, axis=0)

    # Remove channel dimension: (N, 1, L) -> (N, L)
    original = original.squeeze(1)
    reconstructed = reconstructed.squeeze(1)

    n_signals = original.shape[0]

    # Denormalized copies for amplitude-sensitive metrics (Vpp, LAT, NLEO)
    if percentiles is not None:
        p_inf, p_sup = float(percentiles[0]), float(percentiles[1])
        orig_ds  = _denormalize(original,      p_inf, p_sup)
        recon_ds = _denormalize(reconstructed, p_inf, p_sup)
    else:
        orig_ds  = original
        recon_ds = reconstructed

    result = {}

    # MSE — on normalized signals (scale-independent comparison)
    if metrics is None or 'mse' in metrics:
        print(f"  Computing MSE ({n_signals} signals)...")
        result['MSE'] = calculate_mse_batch(original, reconstructed)

    # Vpp error — on original-scale signals
    if metrics is None or 'vpp' in metrics:
        print(f"  Computing Vpp ({n_signals} signals)...")
        result['Vpp_error'] = calculate_vpp_error_batch(orig_ds, recon_ds)

    # DF error — frequency-domain, not affected by amplitude scale
    if metrics is None or 'df' in metrics:
        print(f"  Computing DF ({n_signals} signals)...")
        try:
            result['DF_error'] = calculate_df_error_batch(original, reconstructed, fs=fs)
        except Exception as e:
            print(f"  Warning: DF calculation failed: {e}")

    # NLEO corr (bipolar only) — on original-scale signals
    if is_bipolar and (metrics is None or 'nleo' in metrics):
        try:
            print(f"  Computing NLEO corr ({n_signals} signals)...")
            result['NLEO_corr'] = calculate_nleo_corr_batch(orig_ds, recon_ds)
        except Exception as e:
            print(f"  Warning: NLEO calculation failed: {e}")

    # LAT metrics (unipolar only) — on original-scale signals (sigma_abs_th is mV-calibrated)
    if not is_bipolar and (metrics is None or 'lat' in metrics):
        try:
            print(f"  Computing LAT ({n_signals} signals)...")
            lat_result = calculate_lat_metrics_batch(orig_ds, recon_ds, fs=fs)
            result.update(lat_result)
        except Exception as e:
            print(f"  Warning: LAT calculation failed: {e}")

    return result


# ---------------------------------------------------------------------------
# Argument parser  (minimal – architecture/hyperparams come from MODEL_REGISTRY)
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate pre-trained autoencoder(s) on the test split. '
                    'Architecture and hyperparameters are resolved automatically '
                    'from the .pth filename via MODEL_REGISTRY.'
    )
    parser.add_argument('--pth_path', type=str, nargs='+', required=True,
                        help='Path(s) to .pth model file(s). '
                             'Filename stem must follow the convention '
                             '{UNIPOLAR|BIPOLAR}_{arch_key} (e.g. UNIPOLAR_CLARAE_SCM_GLU.pth).')
    parser.add_argument('--preprocessed_data_dir', type=str, required=True,
                        help='Directory with preprocessed HDF5 data (same as used in training)')
    parser.add_argument('--loss_function', type=str, default=None,
                        choices=['mse', 'dtw'],
                        help='Override loss function (default: read from MODEL_REGISTRY per model)')
    parser.add_argument('--wandb_project', type=str, default='autoencoder-egms-test')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_run_id', type=str, default=None,
                        help='WandB run ID to resume and update (e.g. "abc12345"). '
                             'If set, logs are added to the existing run instead of creating a new one.')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--metrics', nargs='+', default=None,
                        choices=['mse', 'vpp', 'df', 'lat', 'nleo'],
                        help='Clinical metrics to compute (default: all). '
                             'Choices: mse vpp df lat nleo. '
                             'Example: --metrics lat vpp')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Build full args namespace for a single model
# ---------------------------------------------------------------------------

def build_eval_args(pth_stem, cli):
    """
    Resolve model architecture and signal type from pth stem via MODEL_REGISTRY
    and combine with FIXED_PARAMS and CLI overrides into a single args Namespace.

    pth stem convention: {UNIPOLAR|BIPOLAR}_{arch_key}[_{loss}_{noise_tag}]
    e.g. 'UNIPOLAR_CLARAE_SCM_GLU_dtw_noise' → unipolar=True, arch_key='CLARAE_SCM_GLU', loss='dtw'
         'BIPOLAR_ACDAE_mse_noNoise'          → unipolar=False, arch_key='ACDAE', loss='mse'
    """
    stem_upper = pth_stem.upper()
    if stem_upper.startswith("UNIPOLAR_"):
        unipolar = True
        arch_key_raw = pth_stem[len("UNIPOLAR_"):]
    elif stem_upper.startswith("BIPOLAR_"):
        unipolar = False
        arch_key_raw = pth_stem[len("BIPOLAR_"):]
    else:
        raise KeyError(
            f"Cannot determine signal type from '{pth_stem}'.\n"
            f"  Filename stem must start with UNIPOLAR_ or BIPOLAR_."
        )

    arch_key, loss_from_stem, _ = _strip_loss_noise_suffix(arch_key_raw)

    if arch_key not in MODEL_REGISTRY:
        raise KeyError(
            f"'{arch_key}' not found in MODEL_REGISTRY.\n"
            f"  Available keys: {list(MODEL_REGISTRY.keys())}\n"
            f"  Add an entry to MODEL_REGISTRY or rename the .pth file."
        )

    entry = MODEL_REGISTRY[arch_key]
    # entry overrides take precedence over FIXED_PARAMS
    params = {**FIXED_PARAMS, **{k: v for k, v in entry.items()
                                  if k != "model_architecture"}}

    # loss_function priority: CLI override > parsed from filename > fallback 'dtw'
    loss_function = cli.loss_function or loss_from_stem or "dtw"

    return argparse.Namespace(
        model_architecture=entry["model_architecture"],
        unipolar=unipolar,
        # CLI args
        loss_function=loss_function,
        device=cli.device,
        wandb_project=cli.wandb_project,
        wandb_entity=cli.wandb_entity,
        no_wandb=cli.no_wandb,
        # Everything else from FIXED_PARAMS (+ any per-model overrides)
        **params,
    )


# ---------------------------------------------------------------------------
# Data loading (test split only)
# ---------------------------------------------------------------------------

def load_test_data(base_dir, unipolar, percentile_inf=0.05, percentile_sup=99.95, debug_samples=None):
    """Load only the test split. Supports chunked (test_001.h5, …) and legacy (test.h5)."""
    signal_type = 'unipolar' if unipolar else 'bipolar'
    percentile_name = f"p{percentile_inf}_{percentile_sup}"

    pattern = os.path.join(base_dir, f"{signal_type}_*")
    matching_dirs = glob(pattern)
    if len(matching_dirs) == 0:
        raise FileNotFoundError(f"No directory found matching: {pattern}")
    if len(matching_dirs) > 1:
        print(f"[WARNING] Multiple directories found, using: {matching_dirs[0]}")

    data_dir = os.path.join(matching_dirs[0], 'normalized', percentile_name)
    d = load_split_h5(data_dir, 'test', include_signals=True)
    data = d['signals'][:debug_samples] if debug_samples else d['signals']
    percentiles = np.array([d['p_inf'], d['p_sup']])
    test_data = torch.tensor(data, dtype=torch.float32)
    print(f"  Test data loaded: {test_data.shape}")
    print(f"  Percentiles: {percentiles}")
    return test_data, percentiles


# ---------------------------------------------------------------------------
# Evaluate one model
# ---------------------------------------------------------------------------

def evaluate_model(pth_path, args, test_data, percentiles, device):
    pth_stem = os.path.splitext(os.path.basename(pth_path))[0]
    print(f"\n{'='*80}")
    print(f"EVALUATING: {pth_stem}")
    print(f"{'='*80}\n")

    # 0. Init wandb early so the run is visible from the start
    if not args.no_wandb:
        wandb_run_id = getattr(args, 'wandb_run_id', None)
        if wandb_run_id:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                id=wandb_run_id,
                resume='allow',
                reinit=True,
            )
        else:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=pth_stem,
                config=vars(args),
                reinit=True,
            )

    # Build model + criterion
    model, criterion, _ = compile_model(
        args,
        input_channels=1,
        input_length=test_data.shape[2],
    )
    model.to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()
    print(f"  Model loaded from: {pth_path}")

    test_loader = DataLoader(
        TensorDataset(test_data),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    # 1. Clean test
    print("\nEvaluating on CLEAN test data...")
    test_result_clean = validate(model, test_loader, criterion, device,
                                 desc="Test (Clean)", add_noise=False)

    # 2. Noisy test
    print("\nEvaluating on NOISY test data...")
    print(f"  Mode: Random {args.min_noise_types}-{args.max_noise_types} noise types")
    print(f"  SNR range: [{args.noise_snr_min}, {args.noise_snr_max}] dB")
    torch.manual_seed(args.noise_seed)
    np.random.seed(args.noise_seed)
    test_result_noisy = validate(
        model, test_loader, criterion, device,
        desc="Test (Noisy)",
        add_noise=True,
        noise_snr_min=args.noise_snr_min,
        noise_snr_max=args.noise_snr_max,
        noise_types=args.noise_types,
        sampling_freq=args.sampling_freq,
        powerline_freq=args.powerline_freq,
        random_noise=args.random_noise,
        min_noise_types=args.min_noise_types,
        max_noise_types=args.max_noise_types,
        include_clean=False,
    )

    # 3. Extract metrics
    test_loss_clean, test_r2_clean, test_r2_std_clean = test_result_clean
    test_loss_noisy, test_r2_noisy, test_r2_std_noisy = test_result_noisy

    # 4. Print
    print(f"\n{'-'*80}")
    print("TEST RESULTS:")
    print(f"{'-'*80}")
    print(f"Clean Test Loss: {test_loss_clean:.4f}")
    print(f"Clean Test R²:   {test_r2_clean:.4f} ± {test_r2_std_clean:.4f}")
    print(f"\nNoisy Test Loss: {test_loss_noisy:.4f}")
    print(f"Noisy Test R²:   {test_r2_noisy:.4f} ± {test_r2_std_noisy:.4f}")
    print(f"{'-'*80}")

    # 5. Clinical metrics
    print(f"\n{'-'*80}")
    print("CLINICAL METRICS (Original vs Reconstructed)")
    print(f"{'-'*80}")

    is_bipolar = not args.unipolar
    noise_config = {
        'snr_min': args.noise_snr_min,
        'snr_max': args.noise_snr_max,
        'random_noise': True,
        'min_types': args.min_noise_types,
        'max_types': args.max_noise_types,
        'powerline_freq': args.powerline_freq,
    }

    cli_metrics = getattr(args, 'metrics', None)

    print("\n[CLEAN INPUT - Reconstruction Quality]")
    clinical_metrics_clean = calculate_clinical_metrics(
        model, test_data, device, fs=args.sampling_freq,
        is_bipolar=is_bipolar, add_noise=False, metrics=cli_metrics,
        percentiles=percentiles,
    )
    for metric_name, values in clinical_metrics_clean.items():
        print(f"  {metric_name:12s}: {values['mean']:.4f} +/- {values['std']:.4f}")

    print("\n[NOISY INPUT - Denoising Quality]")
    torch.manual_seed(args.noise_seed)
    np.random.seed(args.noise_seed)
    clinical_metrics_noisy = calculate_clinical_metrics(
        model, test_data, device, fs=args.sampling_freq,
        is_bipolar=is_bipolar, add_noise=True, noise_config=noise_config,
        metrics=cli_metrics, percentiles=percentiles,
    )
    for metric_name, values in clinical_metrics_noisy.items():
        print(f"  {metric_name:12s}: {values['mean']:.4f} +/- {values['std']:.4f}")
    print(f"{'-'*80}")

    # 6. Wandb — log results (run was initialised at the top of this function)
    if not args.no_wandb:
        log_dict = {
            "test/loss_clean":     test_loss_clean,
            "test/r2_clean":       test_r2_clean,
            "test/r2_std_clean":   test_r2_std_clean,
            "test/loss_noisy":     test_loss_noisy,
            "test/r2_noisy":       test_r2_noisy,
            "test/r2_std_noisy":   test_r2_std_noisy,
        }
        for metric_name, values in clinical_metrics_clean.items():
            log_dict[f"test/{metric_name}_clean_mean"] = values['mean']
        for metric_name, values in clinical_metrics_noisy.items():
            log_dict[f"test/{metric_name}_noisy_mean"] = values['mean']
        wandb.log(log_dict)

        wandb.run.summary["test_loss_clean"]      = test_loss_clean
        wandb.run.summary["test_r2_clean"]        = test_r2_clean
        wandb.run.summary["test_r2_std_clean"]    = test_r2_std_clean
        wandb.run.summary["test_loss_noisy"]      = test_loss_noisy
        wandb.run.summary["test_r2_noisy"]        = test_r2_noisy
        wandb.run.summary["test_r2_std_noisy"]    = test_r2_std_noisy
        for metric_name, values in clinical_metrics_clean.items():
            wandb.run.summary[f"test_{metric_name}_clean"]      = f"{values['mean']:.4f} ± {values['std']:.4f}"
            wandb.run.summary[f"test_{metric_name}_clean_mean"] = values['mean']
        for metric_name, values in clinical_metrics_noisy.items():
            wandb.run.summary[f"test_{metric_name}_noisy"]      = f"{values['mean']:.4f} ± {values['std']:.4f}"
            wandb.run.summary[f"test_{metric_name}_noisy_mean"] = values['mean']

        # Visualizations
        train_p_inf, train_p_sup = float(percentiles[0]), float(percentiles[1])
        visualize_test_with_noise_info(
            model, test_data, train_p_inf, train_p_sup,
            "test", num_samples=15, args=args, device=device,
            is_bipolar=is_bipolar,
        )

        wandb.finish()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cli = parse_args()
    set_seed(FIXED_PARAMS['seed'])

    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    _data_cache = {}  # keyed by (base_dir, unipolar) to avoid redundant loads

    for pth_path in cli.pth_path:
        if not os.path.isfile(pth_path):
            print(f"\nWARNING: .pth not found, skipping: {pth_path}")
            continue

        pth_stem = os.path.splitext(os.path.basename(pth_path))[0]
        try:
            args = build_eval_args(pth_stem, cli)
        except KeyError as e:
            print(f"\nWARNING: {e}\nSkipping {pth_path}.")
            continue

        cache_key = (cli.preprocessed_data_dir, args.unipolar)
        if cache_key not in _data_cache:
            sig = 'unipolar' if args.unipolar else 'bipolar'
            print(f"\nLoading test data ({sig})...")
            _data_cache[cache_key] = load_test_data(cli.preprocessed_data_dir, args.unipolar)
        test_data, percentiles = _data_cache[cache_key]

        evaluate_model(pth_path, args, test_data, percentiles, device)

    print("\nDone.")


if __name__ == '__main__':
    main()
