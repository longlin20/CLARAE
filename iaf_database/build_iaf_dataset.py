"""
build_iaf_dataset.py
====================
Load the PhysioNet Intracardiac Atrial Fibrillation Database (iaf1–iaf8),
extract bipolar CS leads, resample, segment, normalize and save to an h5 file.

The output h5 is the input for eval_iaf_database.py.

Usage
-----
    python build_iaf_dataset.py \\
        --db_dir  DATABASE/intracardiac-atrial-fibrillation-database-1.0.0 \\
        --ref_h5  processed_data_v2/bipolar_181/normalized/p0.05_99.95/test_001.h5 \\
        --out_h5  iaf_eval_output/iaf_test.h5
"""

import argparse
import os
import sys

import h5py
import numpy as np

# ── rhythm labels per patient (from database headers) ────────────────────────
PATIENT_RHYTHM = {
    'iaf1': 'AF',
    'iaf2': 'AF',
    'iaf3': 'AF',
    'iaf4': 'AF',
    'iaf5': 'AFL',
    'iaf6': 'AF',
    'iaf7': 'AF',
    'iaf8': 'AFL',
}

CS_CHANNELS = {'CS12', 'CS34', 'CS56', 'CS78', 'CS90'}


# ---------------------------------------------------------------------------
# WFDB reader (no external dependency)
# ---------------------------------------------------------------------------

def _parse_hea(hea_path: str):
    """Parse a WFDB .hea file. Returns (fs, n_samples, channels)."""
    channels = []
    fs = None
    n_samples = None
    with open(hea_path, 'r') as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if i == 0 or (fs is None and len(parts) >= 3 and parts[1].isdigit()):
                fs = int(parts[2])
                n_samples = int(parts[3]) if len(parts) > 3 else None
                continue
            if len(parts) >= 9:
                try:
                    gain     = float(parts[2].split('/')[0])
                    baseline = int(parts[4]) if parts[4].lstrip('-').isdigit() else 0
                    name     = parts[8]
                    channels.append({'name': name, 'gain': gain,
                                     'baseline': baseline, 'filename': parts[0]})
                except (ValueError, IndexError):
                    pass
    return fs, n_samples, channels


def _read_dat(dat_path: str, n_channels: int, n_samples: int) -> np.ndarray:
    """Read WFDB format-16 binary file. Returns (n_samples, n_channels) int16."""
    raw = np.frombuffer(open(dat_path, 'rb').read(), dtype='<i2')
    total = n_channels * n_samples
    if len(raw) < total:
        raw = np.concatenate([raw, np.zeros(total - len(raw), dtype='<i2')])
    return raw[:total].reshape(n_samples, n_channels)


def load_wfdb_record(record_path: str):
    """Load one WFDB record. Returns (signals_mV, channel_names, fs)."""
    hea_path = record_path + '.hea'
    dat_path = record_path + '.dat'
    fs, n_samples, channels = _parse_hea(hea_path)
    if n_samples is None:
        raise ValueError(f"Could not parse n_samples from {hea_path}")
    raw = _read_dat(dat_path, len(channels), n_samples)
    signals_mV = np.zeros_like(raw, dtype=np.float32)
    for i, ch in enumerate(channels):
        signals_mV[:, i] = (raw[:, i].astype(np.float32) - ch['baseline']) / ch['gain']
    return signals_mV, [ch['name'] for ch in channels], fs


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

def resample_signal(sig: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(target_fs, orig_fs)
    return resample_poly(sig, target_fs // g, orig_fs // g).astype(np.float32)


def segment_array(sig: np.ndarray, window: int, step: int) -> np.ndarray:
    starts = range(0, len(sig) - window + 1, step)
    return np.stack([sig[s:s + window] for s in starts], axis=0)


def normalize(sig: np.ndarray, p_inf: float, p_sup: float) -> np.ndarray:
    denom = p_sup - p_inf
    if denom == 0:
        return np.zeros_like(sig)
    return np.clip(2.0 * (sig - p_inf) / denom - 1.0, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_iaf_dataset(db_dir: str,
                      p_inf: float, p_sup: float,
                      target_fs: int = 500,
                      window_samples: int = 1250,
                      step_samples: int = 625,
                      skip_seconds: float = 5.0,
                      cs_channels: set = CS_CHANNELS,
                      dc_remove: bool = False):
    """
    Load all WFDB records from the IAF database, extract bipolar CS channels,
    resample, segment, and normalize.

    Parameters
    ----------
    dc_remove : bool
        If True, subtract the per-window mean (in mV) before normalization.
        Removes baseline DC offset so signals are zero-mean, matching training data.
        Recommended for IAF database where recording baselines differ.

    Returns
    -------
    signals     : (N, 1, window_samples) float32 in [-1, 1]
    rhythms     : (N,) str
    patient_ids : (N,) str
    map_names   : (N,) str
    """
    records_file = os.path.join(db_dir, 'RECORDS')
    with open(records_file) as f:
        records = [r.strip() for r in f if r.strip()]

    all_sigs, all_rhy, all_pids, all_mns = [], [], [], []

    for rec_name in records:
        patient_id = rec_name.split('_')[0]
        map_name   = rec_name
        rhythm     = PATIENT_RHYTHM.get(patient_id, 'AF')

        rec_path = os.path.join(db_dir, rec_name)
        try:
            sigs_mV, ch_names, orig_fs = load_wfdb_record(rec_path)
        except Exception as e:
            print(f"  [WARN] Skipping {rec_name}: {e}")
            continue

        cs_idx = [i for i, n in enumerate(ch_names) if n in cs_channels]
        if not cs_idx:
            print(f"  [WARN] No CS channels in {rec_name}  (found: {ch_names})")
            continue

        orig_dur_s        = sigs_mV.shape[0] / orig_fs
        skip_samples_orig = int(skip_seconds * orig_fs)
        sigs_mV           = sigs_mV[skip_samples_orig:]
        usable_dur_s      = sigs_mV.shape[0] / orig_fs
        usable_samples_t  = int(usable_dur_s * target_fs)
        n_windows_exp     = max(0, (usable_samples_t - window_samples) // step_samples + 1)

        n_windows_rec = 0
        for idx in cs_idx:
            ch_sig = sigs_mV[:, idx]
            if orig_fs != target_fs:
                ch_sig = resample_signal(ch_sig, orig_fs, target_fs)
            windows_mv = segment_array(ch_sig, window_samples, step_samples)
            if dc_remove:
                windows_mv = windows_mv - windows_mv.mean(axis=1, keepdims=True)
            windows_norm = np.stack([
                normalize(w, p_inf, p_sup) for w in windows_mv
            ], axis=0)
            if len(windows_norm) == 0:
                continue
            n = len(windows_norm)
            all_sigs.append(windows_norm[:, np.newaxis, :])
            all_rhy.extend([rhythm]      * n)
            all_pids.extend([patient_id] * n)
            all_mns.extend([f"{map_name}_{ch_names[idx]}"] * n)
            n_windows_rec += n

        window_dur_s = window_samples / target_fs
        print(f"  {rec_name:<15}  {orig_dur_s:6.1f}s orig  "
              f"({skip_seconds:.0f}s skipped → {usable_dur_s:.1f}s usable)  "
              f"{len(cs_idx)} CS ch × {n_windows_exp} windows "
              f"({window_dur_s:.1f}s each)  = {n_windows_rec} total")

    if not all_sigs:
        raise RuntimeError("No signals extracted from the IAF database.")

    signals     = np.concatenate(all_sigs, axis=0)
    rhythms     = np.array(all_rhy)
    patient_ids = np.array(all_pids)
    map_names   = np.array(all_mns)

    unique, counts = np.unique(rhythms, return_counts=True)
    print(f"\n  Total: {len(signals):,} windows — "
          + ", ".join(f"{u}:{c:,}" for u, c in zip(unique, counts)))
    print(f"  Patients: {sorted(set(patient_ids.tolist()))}")
    return signals, rhythms, patient_ids, map_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Build IAF h5 dataset from raw WFDB records.')

    p.add_argument('--db_dir',   required=True,
                   help='Path to intracardiac-atrial-fibrillation-database-1.0.0')
    p.add_argument('--out_h5',   required=True,
                   help='Output h5 file path (e.g. iaf_eval_output/iaf_test.h5)')
    p.add_argument('--ref_h5',   default=None,
                   help='Reference h5 to extract p_inf/p_sup (training percentiles). '
                        'If omitted, percentiles are computed from the IAF data itself.')

    p.add_argument('--target_fs',      type=int,   default=500)
    p.add_argument('--window_samples', type=int,   default=1250,
                   help='Window length in samples (default: 1250 = 2.5 s at 500 Hz)')
    p.add_argument('--step_samples',   type=int,   default=1250,
                   help='Step between windows (default: 1250 = no overlap)')
    p.add_argument('--skip_seconds',   type=float, default=0.0,
                   help='Seconds to skip at the start of each recording')
    p.add_argument('--dc_remove',      action='store_true',
                   help='Subtract per-window mean (mV) before normalization to remove DC offset. '
                        'Recommended: fixes baseline shifts in IAF recordings.')

    args = p.parse_args()

    # ── Normalization percentiles ─────────────────────────────────────────────
    if args.ref_h5:
        print(f"Loading normalization from: {args.ref_h5}")
        with h5py.File(args.ref_h5, 'r') as f:
            p_inf = float(f.attrs.get('p_inf_value',
                          f['percentiles'][0] if 'percentiles' in f else -1.0))
            p_sup = float(f.attrs.get('p_sup_value',
                          f['percentiles'][1] if 'percentiles' in f else  1.0))
        print(f"  p_inf={p_inf:.4f}  p_sup={p_sup:.4f}")
    else:
        print("[INFO] No --ref_h5 provided — percentiles will be computed from IAF data.")
        p_inf = p_sup = None

    # ── Build dataset ─────────────────────────────────────────────────────────
    print(f"\nLoading IAF database from: {args.db_dir}")
    if args.dc_remove:
        print(f"[INFO] DC removal: subtracting per-window mean before normalization")

    signals, rhythms, patient_ids, map_names = build_iaf_dataset(
        db_dir         = args.db_dir,
        p_inf          = p_inf if p_inf is not None else 0.0,
        p_sup          = p_sup if p_sup is not None else 1.0,
        target_fs      = args.target_fs,
        window_samples = args.window_samples,
        step_samples   = args.step_samples,
        skip_seconds   = args.skip_seconds,
        dc_remove      = args.dc_remove,
    )

    if p_inf is None:
        print("Computing percentiles from IAF signals ...")
        flat  = signals.reshape(-1)
        p_inf = float(np.percentile(flat, 0.05))
        p_sup = float(np.percentile(flat, 99.95))
        print(f"  p_inf={p_inf:.4f}  p_sup={p_sup:.4f}")
        scale   = p_sup - p_inf
        signals = np.clip(2.0 * (signals - p_inf) / scale - 1.0, -1.0, 1.0
                          ).astype(np.float32)

    # ── Save h5 ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out_h5)), exist_ok=True)
    with h5py.File(args.out_h5, 'w') as f:
        f.create_dataset('signals',     data=signals,
                         compression='gzip', compression_opts=4)
        f.create_dataset('rhythms',
                         data=np.array(rhythms.tolist(),     dtype='S'))
        f.create_dataset('patient_ids',
                         data=np.array(patient_ids.tolist(), dtype='S'))
        f.create_dataset('map_names',
                         data=np.array(map_names.tolist(),   dtype='S'))
        f.attrs['p_inf_value']    = p_inf
        f.attrs['p_sup_value']    = p_sup
        f.attrs['source']         = 'intracardiac-atrial-fibrillation-database'
        f.attrs['target_fs']      = args.target_fs
        f.attrs['window_samples'] = args.window_samples
        f.attrs['step_samples']   = args.step_samples
        f.attrs['dc_remove']      = int(args.dc_remove)

    print(f"\nGuardado: {args.out_h5}  ({len(signals):,} ventanas)")


if __name__ == '__main__':
    main()
