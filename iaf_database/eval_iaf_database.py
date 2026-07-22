"""
eval_iaf_database.py
====================
Evaluate autoencoder reconstruction and denoising on a pre-built IAF h5 dataset.
Build the h5 first with build_iaf_dataset.py.

Model architecture is auto-detected from the .pth filename via MODEL_REGISTRY.

Usage
-----
    # Single model
    python eval_iaf_database.py \\
        --iaf_h5   iaf_eval_output/iaf_test.h5 \\
        --pth_path model_pth/bipolar/clarae/BIPOLAR_CLARAE_SCM_GLU_AP_ELU_dtw_noise.pth

    # All models in a directory
    python eval_iaf_database.py \\
        --iaf_h5   iaf_eval_output/iaf_test.h5 \\
        --pth_dir  model_pth/bipolar/clarae \\
        --wandb_project autoencoder-egms-iaf
"""

import argparse
import contextlib
import glob as _glob
import io
import os
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from eval_utils import load_model
from utils import add_combined_noise
from plots.plot_AE_results import visualize_test_with_noise_info
from metrics.Vpp import calculate_vpp_batch
from metrics.DF import calculate_df_batch
from functions.NLEO_functions import calculateNLEORaw
from run_test import build_eval_args

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _reconstruct(model, signals: np.ndarray, batch_size: int, device) -> np.ndarray:
    model.eval()
    recons = []
    tensor = torch.from_numpy(signals)
    for (batch,) in DataLoader(TensorDataset(tensor), batch_size=batch_size):
        recons.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(recons, axis=0)


def _clinical_metrics(orig, recon, patient_ids, p_inf, p_sup, is_bipolar, fs):
    """
    Compute R², Vpp error, DF error and NLEO corr (bipolar) per signal,
    looping patient by patient to avoid OOM on large datasets.

    Returns
    -------
    patient_rows : list of dicts, one per patient (nanmean of signal metrics)
    summary      : dict with mean ± std across ALL signals
    """
    orig_2d  = orig.squeeze(1)
    recon_2d = recon.squeeze(1)
    n_total  = len(orig_2d)
    scale    = (p_sup - p_inf) if (p_inf is not None and p_sup is not None) else None

    pids_unique = sorted(np.unique(patient_ids))
    n_pids      = len(pids_unique)

    all_r2  = []; all_vpp = []; all_df  = []
    all_nleo = [] if is_bipolar else None

    patient_rows = []

    for idx, pid in enumerate(pids_unique):
        mask = patient_ids == pid
        o    = orig_2d[mask]
        r    = recon_2d[mask]

        if scale is not None:
            o_ds = (o + 1) / 2 * scale + p_inf
            r_ds = (r + 1) / 2 * scale + p_inf
        else:
            o_ds, r_ds = o, r

        prefix = f"  [{idx+1}/{n_pids}] {pid} ({mask.sum()} señales)"

        # ── R² per signal ────────────────────────────────────────────────────
        print(f"{prefix} — R² ...", flush=True)
        o_f    = o.reshape(len(o), -1)
        r_f    = r.reshape(len(r), -1)
        ss_res = ((o_f - r_f) ** 2).sum(axis=1)
        ss_tot = ((o_f - o_f.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        r2_p   = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)

        # ── Vpp per signal ───────────────────────────────────────────────────
        print(f"{prefix} — Vpp ...", flush=True)
        vpp_p = np.abs(calculate_vpp_batch(o_ds) - calculate_vpp_batch(r_ds))

        # ── DF per signal ────────────────────────────────────────────────────
        print(f"{prefix} — DF ...", flush=True)
        df_p = np.full(len(o), float('nan'))
        try:
            df_o = calculate_df_batch(o, fs=fs, verbose=False, show_progress=False)
            df_r = calculate_df_batch(r, fs=fs, verbose=False, show_progress=False)
            df_p = np.abs(np.array(df_o.DF_values) - np.array(df_r.DF_values)).astype(float)
        except Exception:
            pass

        row = {
            'patient':   pid,
            'n_signals': int(mask.sum()),
            'r2':        float(np.nanmean(r2_p)),
            'vpp_error': float(np.nanmean(vpp_p)),
            'df_error':  float(np.nanmean(df_p)),
        }

        # ── NLEO corr per signal (bipolar) ───────────────────────────────────
        if is_bipolar:
            print(f"{prefix} — NLEO ...", flush=True)
            nleo_p = np.full(len(o), float('nan'))
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    nleo_o = calculateNLEORaw(o_ds)
                    nleo_r = calculateNLEORaw(r_ds)
                vals   = np.array([
                    np.corrcoef(nleo_o[i], nleo_r[i])[0, 1]
                    for i in range(len(o))
                ])
                nleo_p = np.nan_to_num(vals, nan=0.0)
            except Exception:
                pass
            row['nleo_corr'] = float(np.nanmean(nleo_p))
            all_nleo.extend(nleo_p.tolist())

        print(f"{prefix} — listo  r2={row['r2']:.4f}  vpp={row['vpp_error']:.4f}"
              f"  df={row['df_error']:.4f}", flush=True)

        all_r2.extend(r2_p.tolist())
        all_vpp.extend(vpp_p.tolist())
        all_df.extend(df_p.tolist())
        patient_rows.append(row)

    # ── Summary: mean ± std across ALL signals ────────────────────────────────
    all_r2 = np.array(all_r2); all_vpp = np.array(all_vpp); all_df = np.array(all_df)
    summary = {
        'r2_mean':        float(np.nanmean(all_r2)),
        'r2_std':         float(np.nanstd(all_r2)),
        'vpp_error_mean': float(np.nanmean(all_vpp)),
        'vpp_error_std':  float(np.nanstd(all_vpp)),
        'df_error_mean':  float(np.nanmean(all_df)),
        'df_error_std':   float(np.nanstd(all_df)),
    }
    if is_bipolar:
        a = np.array(all_nleo)
        summary['nleo_corr_mean'] = float(np.nanmean(a))
        summary['nleo_corr_std']  = float(np.nanstd(a))

    print(f"\n  Total {n_total} señales — "
          f"r2={summary['r2_mean']:.4f}±{summary['r2_std']:.4f}  "
          f"vpp={summary['vpp_error_mean']:.4f}±{summary['vpp_error_std']:.4f}  "
          f"df={summary['df_error_mean']:.4f}±{summary['df_error_std']:.4f}", flush=True)

    return patient_rows, summary


def _log_wandb_table(rows, key):
    if not (WANDB_AVAILABLE and wandb.run is not None) or not rows:
        return
    columns = list(rows[0].keys())
    data    = [[row.get(c, float('nan')) for c in columns] for row in rows]
    wandb.log({key: wandb.Table(columns=columns, data=data)})


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_one(pth_path, signals, rhythms, patient_ids, map_names,
                 p_inf, p_sup, target_fs, args, device):
    """Evaluate a single .pth model on pre-loaded IAF signals."""
    pth_stem = os.path.splitext(os.path.basename(pth_path))[0]

    cli_ns = argparse.Namespace(
        loss_function  = None,
        device         = str(device),
        wandb_project  = args.wandb_project,
        wandb_entity   = getattr(args, 'wandb_entity', None),
        no_wandb       = args.no_wandb,
    )
    try:
        eval_ns     = build_eval_args(pth_stem, cli_ns)
        model_class = eval_ns.model_architecture
        is_bipolar  = not eval_ns.unipolar
    except KeyError as e:
        print(f"  [WARN] Could not auto-detect model from '{pth_stem}': {e}")
        return {}

    print(f"\nLoading model: {pth_stem}")
    model = load_model(
        pth_path        = pth_path,
        model_class     = model_class,
        latent_dim      = eval_ns.latent_dim,
        filters_initial = eval_ns.filters_initial,
        dropout_rate    = eval_ns.dropout_rate,
        dense_dim       = eval_ns.dense_dim,
        input_length    = signals.shape[2],
        device          = str(device),
        q_parameter     = eval_ns.q_parameter,
    )
    print(f"  Architecture : {model_class}")
    print(f"  Signal type  : {'bipolar' if is_bipolar else 'unipolar'}")
    print(f"  Input shape  : {signals.shape}")

    if not args.no_wandb and WANDB_AVAILABLE:
        wandb.init(
            project = args.wandb_project,
            entity  = getattr(args, 'wandb_entity', None),
            name    = f"{pth_stem}_iaf",
            config  = {**vars(args), 'model_class': model_class,
                       'is_bipolar': is_bipolar},
        )

    log_dict = {}

    # ── Reconstruction ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Reconstruction (clean input, {len(signals):,} señales) ...")
    recons = _reconstruct(model, signals, args.batch_size, device)
    patient_rows_clean, summary_clean = _clinical_metrics(
        signals, recons, patient_ids, p_inf, p_sup, is_bipolar, target_fs)

    print(f"\n  {'Patient':<10} {'N':>6} {'R²':>8} {'Vpp_err':>10} {'DF_err':>10}"
          + (' nleo_corr' if is_bipolar else ''))
    for row in patient_rows_clean:
        extra = f"  {row['nleo_corr']:>8.4f}" if is_bipolar else ''
        print(f"  {row['patient']:<10} {row['n_signals']:>6,} {row['r2']:>8.4f} "
              f"{row['vpp_error']:>10.4f} {row['df_error']:>10.4f}{extra}")

    log_dict.update({f'recon/{k}': v for k, v in summary_clean.items()})

    # ── Denoising ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Denoising  (SNR [{args.noise_snr_min}, {args.noise_snr_max}] dB) ...")
    torch.manual_seed(args.noise_seed)
    np.random.seed(args.noise_seed)
    sigs_tensor  = torch.from_numpy(signals).float()
    noisy_tensor = add_combined_noise(
        sigs_tensor,
        noise_types = ['gaussian', 'baseline_wander', 'powerline', 'spike'],
        snr_db_min  = args.noise_snr_min,
        snr_db_max  = args.noise_snr_max,
        fs          = target_fs,
    )
    denoised = _reconstruct(model, noisy_tensor.numpy(), args.batch_size, device)

    viz_args = argparse.Namespace(
        noise_snr_min  = args.noise_snr_min,
        noise_snr_max  = args.noise_snr_max,
        sampling_freq  = target_fs,
        powerline_freq = 50,
    )
    visualize_test_with_noise_info(
        model, sigs_tensor, p_inf, p_sup,
        label       = 'test',
        num_samples = args.num_viz_samples,
        args        = viz_args,
        device      = str(device),
        is_bipolar  = is_bipolar,
    )

    patient_rows_dn, summary_dn = _clinical_metrics(
        signals, denoised, patient_ids, p_inf, p_sup, is_bipolar, target_fs)

    print(f"\n  {'Patient':<10} {'N':>6} {'R²':>8} {'Vpp_err':>10} {'DF_err':>10}"
          + (' nleo_corr' if is_bipolar else ''))
    for row in patient_rows_dn:
        extra = f"  {row['nleo_corr']:>8.4f}" if is_bipolar else ''
        print(f"  {row['patient']:<10} {row['n_signals']:>6,} {row['r2']:>8.4f} "
              f"{row['vpp_error']:>10.4f} {row['df_error']:>10.4f}{extra}")

    log_dict.update({f'denoise/{k}': v for k, v in summary_dn.items()})

    # ── Summary ───────────────────────────────────────────────────────────────
    def _fmt(m, s): return f"{m:.4f} ± {s:.4f}"
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  Model    : {pth_stem}")
    print(f"  Signals  : {len(signals):,}")
    print(f"  {'Metric':<22}  {'Recon':>16}  {'Denoised':>16}")
    print(f"  {'R²':<22}  {_fmt(summary_clean['r2_mean'], summary_clean['r2_std']):>16}"
          f"  {_fmt(summary_dn['r2_mean'], summary_dn['r2_std']):>16}")
    print(f"  {'Vpp error':<22}  {_fmt(summary_clean['vpp_error_mean'], summary_clean['vpp_error_std']):>16}"
          f"  {_fmt(summary_dn['vpp_error_mean'], summary_dn['vpp_error_std']):>16}")
    print(f"  {'DF error':<22}  {_fmt(summary_clean['df_error_mean'], summary_clean['df_error_std']):>16}"
          f"  {_fmt(summary_dn['df_error_mean'], summary_dn['df_error_std']):>16}")
    if is_bipolar:
        print(f"  {'NLEO corr':<22}  {_fmt(summary_clean['nleo_corr_mean'], summary_clean['nleo_corr_std']):>16}"
              f"  {_fmt(summary_dn['nleo_corr_mean'], summary_dn['nleo_corr_std']):>16}")

    # ── WandB ─────────────────────────────────────────────────────────────────
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(log_dict)
        for k, v in log_dict.items():
            wandb.run.summary[k] = v
        _log_wandb_table(patient_rows_clean, 'recon/per_patient')
        _log_wandb_table(patient_rows_dn,    'denoise/per_patient')
        wandb.finish()

    return log_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device(
        args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')

    if args.pth_dir:
        pth_paths = sorted(_glob.glob(os.path.join(args.pth_dir, '*.pth')))
        if not pth_paths:
            raise FileNotFoundError(f"No .pth files found in: {args.pth_dir}")
        print(f"Found {len(pth_paths)} model(s) in {args.pth_dir}")
    else:
        pth_paths = [args.pth_path]

    # ── Load pre-built h5 ─────────────────────────────────────────────────────
    print(f"\nLoading IAF dataset from: {args.iaf_h5}")
    with h5py.File(args.iaf_h5, 'r') as f:
        signals     = f['signals'][:]
        rhythms     = np.array([r.decode() for r in f['rhythms'][:]])
        patient_ids = np.array([p.decode() for p in f['patient_ids'][:]])
        map_names   = np.array([m.decode() for m in f['map_names'][:]])
        p_inf       = float(f.attrs['p_inf_value'])
        p_sup       = float(f.attrs['p_sup_value'])
        target_fs   = int(f.attrs.get('target_fs', 500))

    print(f"  Signals : {len(signals):,}  shape={signals.shape}")
    print(f"  p_inf={p_inf:.4f}  p_sup={p_sup:.4f}  fs={target_fs} Hz")
    unique, counts = np.unique(rhythms, return_counts=True)
    print(f"  Rhythms : " + ", ".join(f"{u}:{c:,}" for u, c in zip(unique, counts)))
    print(f"  Patients: {sorted(set(patient_ids.tolist()))}")

    all_results = {}
    for pth_path in pth_paths:
        result = evaluate_one(
            pth_path    = pth_path,
            signals     = signals,
            rhythms     = rhythms,
            patient_ids = patient_ids,
            map_names   = map_names,
            p_inf       = p_inf,
            p_sup       = p_sup,
            target_fs   = target_fs,
            args        = args,
            device      = device,
        )
        pth_stem = os.path.splitext(os.path.basename(pth_path))[0]
        all_results[pth_stem] = result

    return all_results


def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate AE reconstruction/denoising on pre-built IAF h5 dataset.')

    p.add_argument('--iaf_h5', required=True,
                   help='Path to h5 built by build_iaf_dataset.py')

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--pth_path', type=str,
                     help='Path to a single AE .pth checkpoint')
    grp.add_argument('--pth_dir',  type=str,
                     help='Directory containing .pth checkpoints (all evaluated)')

    p.add_argument('--noise_snr_min',   type=float, default=-5.0)
    p.add_argument('--noise_snr_max',   type=float, default=10.0)
    p.add_argument('--noise_seed',      type=int,   default=42)
    p.add_argument('--num_viz_samples', type=int,   default=15)
    p.add_argument('--batch_size',      type=int,   default=256)
    p.add_argument('--device',          type=str,   default='cuda')
    p.add_argument('--wandb_project',   type=str,   default='autoencoder-egms-iaf')
    p.add_argument('--wandb_entity',    type=str,   default=None)
    p.add_argument('--no_wandb',        action='store_true')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
