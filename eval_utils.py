"""
eval_utils.py
=============
Shared utilities for EGM autoencoder evaluation pipeline.

Provides:
  - Constants / registries  (RHYTHM_COLORS, MODEL_REGISTRY, …)
  - MLPClassifier class
  - Path-resolution helpers (_signal_type_from_stem, _find_dataset_path, …)
  - Model helpers           (build_model, load_model, …)
  - Dataset helpers         (load_test_dataset, extract_latent, …)

Imported by: latent_space_analysis.py, eval_pipeline.py, miss_classifier.py,
             classify_from_pth.py
"""

import os
import re
import importlib
import json
from glob import glob
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import h5py
from torch.utils.data import DataLoader, TensorDataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed. WandB logging will be skipped.")

try:
    import plotly.graph_objects as go  # noqa: F401
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Colour palette (consistent across plots)
# ---------------------------------------------------------------------------
RHYTHM_COLORS = {
    'AF':       '#DC2626',
    'SR300':    '#2563EB',
    'SR600':    '#059669',
    'Flutter':  '#EA580C',
    'SR':       '#CA8A04',
    'Other_SR': '#0891B2',
    'unknown':  '#AAAAAA',
}
RHYTHM_PRIORITY  = ['AF', 'Flutter', 'SR300', 'SR600', 'Other_SR', 'SR']
EXCLUDED_RHYTHMS = {'not_reliable', 'TA', 'insufficient', 'unclassified', 'unknown'}

# ---------------------------------------------------------------------------
# Model registry (centralizado en model_registry.py)
# ---------------------------------------------------------------------------
from model_registry import (
    MODEL_REGISTRY, SC_MODELS, _CLARAE_MODELS, ENCODE_MODELS,
    _KNOWN_LOSSES, _KNOWN_NOISE_TAGS,
)


# ---------------------------------------------------------------------------
# MLP Classifier
# ---------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    """1-hidden-layer MLP classifier."""
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class _ResBlock(nn.Module):
    """Pre-activation residual block: BN → GELU → Linear → BN → GELU → Linear + skip."""
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class DeepMLPClassifier(nn.Module):
    """
    Deeper residual MLP: input_proj → N residual blocks → bottleneck → output.
    Architecture: input → hidden (256) →[ResBlock]×n_blocks → hidden//2 → n_classes
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, n_classes: int = 6,
                 dropout: float = 0.4, n_blocks: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.Sequential(
            *[_ResBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x):
        return self.head(self.res_blocks(self.input_proj(x)))


# ---------------------------------------------------------------------------
# Path-resolution helpers
# ---------------------------------------------------------------------------

from model_registry import _strip_loss_noise_suffix  # noqa: E402


def _arch_from_stem(pth_stem: str) -> str:
    """
    Infer model_class from a pth filename stem.
    Convention: {UNIPOLAR|BIPOLAR}_{ARCH_KEY}[_{loss}_{noise_tag}]
    e.g. 'UNIPOLAR_CLARAE_SCM_GLU_dtw_noise' -> 'CLARAE_SCM_GLU'
    """
    stem_upper = pth_stem.upper()
    if stem_upper.startswith("UNIPOLAR_"):
        arch_key_raw = pth_stem[len("UNIPOLAR_"):]
    elif stem_upper.startswith("BIPOLAR_"):
        arch_key_raw = pth_stem[len("BIPOLAR_"):]
    else:
        raise ValueError(
            f"Cannot infer model_class from '{pth_stem}'. "
            f"Stem must start with UNIPOLAR_ or BIPOLAR_."
        )
    arch_key, _, _ = _strip_loss_noise_suffix(arch_key_raw)
    if arch_key not in MODEL_REGISTRY:
        raise ValueError(
            f"'{arch_key}' not found in MODEL_REGISTRY. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return arch_key


def _signal_type_from_stem(pth_stem: str) -> str:
    """Returns 'unipolar' or 'bipolar' inferred from pth filename stem."""
    stem_upper = pth_stem.upper()
    if stem_upper.startswith("UNIPOLAR_"):
        return "unipolar"
    elif stem_upper.startswith("BIPOLAR_"):
        return "bipolar"
    else:
        raise ValueError(
            f"Cannot infer signal type from '{pth_stem}'. "
            f"Stem must start with UNIPOLAR_ or BIPOLAR_."
        )


def _find_dataset_path(data_dir: str, signal_type: str) -> str:
    """
    Search data_dir for a subdirectory matching:
        {signal_type}_*/normalized/*/
    that contains test split files (chunked test_001.h5 or legacy test.h5).
    """
    for test_fn in ('test_001.h5', 'test.h5'):
        pattern = os.path.join(data_dir, f"{signal_type}_*", "normalized", "*", test_fn)
        matches = sorted(glob(pattern))
        if matches:
            if len(matches) > 1:
                print(f"  [auto-detect] Multiple '{signal_type}' datasets found, using first:")
                for m in matches:
                    print(f"    {m}")
            return os.path.dirname(matches[0])
    raise FileNotFoundError(
        f"No test data found for '{signal_type}' under '{data_dir}'.\n"
        f"Expected: test_001.h5 (chunked) or test.h5 (legacy)"
    )


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def build_model(model_class: str, latent_dim: int, filters_initial: int,
                dropout_rate: float, dense_dim: int, input_length: int,
                input_channels: int = 1, q_parameter: int = 2):
    """Instantiate an autoencoder model by class name."""
    if model_class not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_class '{model_class}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    module_path, class_name = MODEL_REGISTRY[model_class]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    if model_class in _CLARAE_MODELS:
        model = cls(
            input_channels=input_channels,
            input_length=input_length,
            latent_dim=latent_dim,
            filters_initial=filters_initial,
            dropout_rate=dropout_rate,
            dense_dim=dense_dim,
        )
    elif model_class == 'DRNN':
        model = cls(input_length=input_length, hidden_size=latent_dim,
                    dropout_rate=dropout_rate)
    elif model_class in ('CNN-DAE', 'FCN-DAE'):
        model = cls(input_length=input_length)
    elif model_class == 'DEEP-FILTER':
        model = cls(input_channels=input_channels)
    elif model_class == 'ACDAE':
        model = cls(input_channels=input_channels, signal_size=input_length)
    elif model_class == 'FGDAE':
        model = cls(signal_size=input_length, q=q_parameter)
    else:
        raise ValueError(f"No instantiation rule for model_class '{model_class}'.")

    return model


def _infer_hyperparams_from_state(state: dict, model_class: str,
                                   latent_dim: int, filters_initial: int,
                                   dense_dim: int, input_length: int):
    """
    Auto-detect dense_dim, latent_dim, and filters_initial from the state dict
    so the model is built with the exact same dimensions as the checkpoint.
    Falls back to the user-supplied values if keys are not found.
    """
    if 'encoder.fc1.weight' in state:
        inferred_dense = state['encoder.fc1.weight'].shape[0]
        if inferred_dense != dense_dim:
            print(f"  [auto-detect] dense_dim: {dense_dim} -> {inferred_dense}")
            dense_dim = inferred_dense

    # VAE encoder uses fc_mu/fc_logvar instead of fc2
    _latent_key = 'encoder.fc2.weight' if 'encoder.fc2.weight' in state \
                  else 'encoder.fc_mu.weight'
    if _latent_key in state:
        inferred_latent = state[_latent_key].shape[0]
        if inferred_latent != latent_dim:
            print(f"  [auto-detect] latent_dim: {latent_dim} -> {inferred_latent}")
            latent_dim = inferred_latent

    if 'encoder.conv1.weight' in state:
        inferred_fi = state['encoder.conv1.weight'].shape[0]
        if inferred_fi != filters_initial:
            print(f"  [auto-detect] filters_initial: {filters_initial} -> {inferred_fi}")
            filters_initial = inferred_fi

    return latent_dim, filters_initial, dense_dim, input_length


def load_model(pth_path: str, model_class: str, latent_dim: int,
               filters_initial: int, dropout_rate: float, dense_dim: int,
               input_length: int, device: str = 'cpu', q_parameter: int = 2):
    """Load a pretrained autoencoder from a .pth checkpoint."""
    state = torch.load(pth_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']

    latent_dim, filters_initial, dense_dim, input_length = \
        _infer_hyperparams_from_state(state, model_class, latent_dim,
                                      filters_initial, dense_dim, input_length)

    model = build_model(model_class, latent_dim, filters_initial,
                        dropout_rate, dense_dim, input_length,
                        q_parameter=q_parameter)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _decode_str_array(arr):
    """Decode a numpy array of bytes/str to a plain str numpy array."""
    return np.array([
        v.decode('utf-8') if isinstance(v, bytes) else v for v in arr
    ])


# ---------------------------------------------------------------------------
# Unified h5 loader  (chunked format: {split}_001.h5 … OR legacy {split}.h5)
# ---------------------------------------------------------------------------

def _load_one_h5(h5_path: str, split: str,
                 include_signals: bool, include_map_names: bool) -> dict:
    """Read one h5 file and return a raw result dict (no filtering)."""
    with h5py.File(h5_path, 'r') as f:
        signals = f[f'{split}_data'][:] if include_signals else None

        rhythms = _decode_str_array(f[f'{split}_rhythms'][:]) \
            if f'{split}_rhythms' in f else None

        patient_ids = _decode_str_array(f[f'{split}_patient_ids'][:]) \
            if f'{split}_patient_ids' in f else None

        mn_key = f'{split}_map_names'
        map_names = _decode_str_array(f[mn_key][:]) \
            if (include_map_names and mn_key in f) else None

        if 'p_inf_value' in f.attrs:
            p_inf = float(f.attrs['p_inf_value'])
            p_sup = float(f.attrs['p_sup_value'])
        elif 'percentiles' in f:
            p_inf = float(f['percentiles'][0])
            p_sup = float(f['percentiles'][1])
        else:
            p_inf, p_sup = None, None

    return dict(signals=signals, rhythms=rhythms, patient_ids=patient_ids,
                map_names=map_names, p_inf=p_inf, p_sup=p_sup)


def load_split_h5(dataset_path: str, split: str,
                  include_signals: bool = True,
                  include_map_names: bool = False) -> dict:
    """
    Unified h5 loader.  Handles both:
      Chunked format : {split}_001.h5, {split}_002.h5, …  (10 patients each)
      Legacy format  : {split}.h5

    Parameters
    ----------
    include_signals   : load the heavy signal array; set False for metadata-only.
    include_map_names : also load {split}_map_names when available.

    Returns
    -------
    dict with keys:
      'signals'     : np.ndarray (N,1,L) or None
      'rhythms'     : np.ndarray (N,)
      'patient_ids' : np.ndarray (N,) or None
      'map_names'   : np.ndarray (N,) or None
      'p_inf'       : float or None
      'p_sup'       : float or None
    """
    chunk_re = re.compile(rf'^{re.escape(split)}_(\d+)\.h5$')
    entries = sorted(os.listdir(dataset_path)) if os.path.isdir(dataset_path) else []
    chunk_files = [
        os.path.join(dataset_path, fn)
        for fn in entries if chunk_re.match(fn)
    ]

    if chunk_files:
        parts = [_load_one_h5(f, split, include_signals, include_map_names)
                 for f in chunk_files]

        def _cat(key):
            arrays = [p[key] for p in parts if p[key] is not None]
            return np.concatenate(arrays, axis=0) if arrays else None

        return dict(
            signals     = _cat('signals'),
            rhythms     = _cat('rhythms'),
            patient_ids = _cat('patient_ids'),
            map_names   = _cat('map_names'),
            p_inf       = parts[0]['p_inf'],
            p_sup       = parts[0]['p_sup'],
        )

    # ── Legacy single-file ───────────────────────────────────────────────────
    h5_path = os.path.join(dataset_path, f'{split}.h5')
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(
            f"No h5 files found for split '{split}' in: {dataset_path}\n"
            f"  Looked for: {split}_001.h5 (chunked) or {split}.h5 (legacy)"
        )
    return _load_one_h5(h5_path, split, include_signals, include_map_names)


def _load_rhythm_map_from_json(json_dir: str) -> dict:
    """
    Read all JSON metadata files in json_dir and build
        patient_id -> dominant_rhythm
    Priority: AF > Flutter > SR300 > SR600 > Other_SR > RS
    """
    patient_rhythm = {}
    json_files = sorted(glob(os.path.join(json_dir, '*.json')))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in '{json_dir}'.")

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        for patient_id, maps in data.items():
            if not isinstance(maps, dict):
                continue
            counts = Counter()
            for k, v in maps.items():
                if k == 'done' or not isinstance(v, dict):
                    continue
                rq = v.get('rhythm_quality', '')
                if rq and rq not in EXCLUDED_RHYTHMS:
                    counts[rq] += 1
            dominant = next(
                (r for r in RHYTHM_PRIORITY if counts.get(r, 0) > 0), None
            )
            if dominant is None and counts:
                dominant = counts.most_common(1)[0][0]
            if dominant:
                patient_rhythm[patient_id] = dominant
    return patient_rhythm


def load_test_dataset(dataset_path: str, n_signals_per_class: int,
                      json_dir: str = None, split: str = 'test', seed: int = 42):
    """
    Load signals and rhythm labels from the HDF5 file.

    Returns
    -------
    signals  : np.ndarray  (N, 1, signal_length)
    rhythms  : np.ndarray  (N,)  string labels
    p_inf, p_sup : float or None
    """
    print(f"\n{'='*60}")
    print(f"Loading '{split}' split from: {dataset_path}")
    rng = np.random.RandomState(seed)

    data    = load_split_h5(dataset_path, split, include_signals=True)
    all_signals = data['signals']
    rhythms     = data['rhythms']
    patient_ids = data['patient_ids']
    p_inf       = data['p_inf']
    p_sup       = data['p_sup']

    if rhythms is not None:
        print(f"  Rhythms: read from H5 (per-map labels)")
    else:
        if patient_ids is None:
            raise KeyError(
                f"Neither '{split}_rhythms' nor '{split}_patient_ids' found in H5."
            )
        if not json_dir:
            raise ValueError(
                f"H5 has no '{split}_rhythms' key. "
                "Pass --json_dir pointing to the egms_data directory, "
                "or re-run preprocess_database.py with --json_dir."
            )
        rhythm_map = _load_rhythm_map_from_json(json_dir)
        rhythms    = np.array([rhythm_map.get(pid, 'unknown') for pid in patient_ids])
        print(f"  Rhythms: derived from JSON in '{json_dir}' (patient-level fallback)")

    rhythms = np.where(rhythms == 'RS', 'SR', rhythms)

    valid_mask  = np.array([r not in EXCLUDED_RHYTHMS for r in rhythms])
    all_signals = all_signals[valid_mask]
    rhythms     = rhythms[valid_mask]

    unique, counts = np.unique(rhythms, return_counts=True)
    print(f"\nClass distribution (before sampling):")
    for u, c in zip(unique, counts):
        print(f"  {u:>12s}: {c:>7,}")

    if n_signals_per_class is None:
        print(f"\nUsing all {len(rhythms):,} signals (no sampling cap).")
        return all_signals, rhythms, p_inf, p_sup

    selected_idx = []
    for rhythm in unique:
        idx = np.where(rhythms == rhythm)[0]
        n   = min(n_signals_per_class, len(idx))
        chosen = rng.choice(idx, n, replace=False)
        selected_idx.append(chosen)

    selected_idx = np.concatenate(selected_idx)
    rng.shuffle(selected_idx)

    signals_out = all_signals[selected_idx]
    rhythms_out = rhythms[selected_idx]

    unique2, counts2 = np.unique(rhythms_out, return_counts=True)
    print(f"\nClass distribution (after sampling, max {n_signals_per_class}/class):")
    for u, c in zip(unique2, counts2):
        print(f"  {u:>12s}: {c:>7,}")
    print(f"  Total: {len(rhythms_out):,} signals")

    return signals_out, rhythms_out, p_inf, p_sup


# ---------------------------------------------------------------------------
# Latent space extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_latent(model, signals: np.ndarray, model_class: str,
                   batch_size: int, device: str, concat_sigma: bool = False,
                   k_samples: int = 1):
    """
    Run the encoder on all signals and return the latent vectors as numpy.

    Parameters
    ----------
    signals      : (N, 1, L)  float32 array (already normalised)
    concat_sigma : If True and the model is a VAE (encoder returns mu + log_var),
                   returns np.concatenate([mu, sigma], axis=1) where
                   sigma = exp(0.5 * log_var).  Doubles the feature dimension.
                   Has no effect on non-VAE models.  Ignored when k_samples > 1.
    k_samples    : For VAE models only.  If > 1, temporarily switch to train mode
                   so reparameterize draws stochastic samples, run the encoder K
                   times, and return the mean z.  k_samples=1 (default) returns
                   mu deterministically (eval mode behaviour).
    """
    use_skip   = model_class in SC_MODELS
    use_encode = model_class in ENCODE_MODELS
    latents    = []

    # For VAE K-sampling we need stochastic reparameterize (train mode)
    _is_vae_sample = k_samples > 1 and use_skip
    if _is_vae_sample:
        _was_training = model.training
        model.train()

    tensor  = torch.from_numpy(signals.astype(np.float32))
    dataset = DataLoader(TensorDataset(tensor), batch_size=batch_size,
                         shuffle=False, num_workers=0)

    for (batch,) in dataset:
        batch = batch.to(device)
        if use_encode:
            z = model.encode(batch)
            latents.append(z.cpu().numpy())
        elif use_skip:
            if _is_vae_sample:
                # Sample K stochastic z's and average them
                z_list = []
                for _ in range(k_samples):
                    out = model.encoder(batch)
                    z_list.append(out[0])   # out[0] is the reparameterized z
                z_avg = torch.stack(z_list, dim=0).mean(dim=0)
                latents.append(z_avg.cpu().numpy())
            else:
                out = model.encoder(batch)
                # VAE encoders return (z, mu, log_var, skips); standard SC encoders return (z, skips)
                if concat_sigma and isinstance(out, (tuple, list)) and len(out) >= 3 \
                        and isinstance(out[2], torch.Tensor) and out[2].shape == out[1].shape:
                    mu_b    = out[1]
                    sigma_b = torch.exp(0.5 * out[2])
                    latents.append(torch.cat([mu_b, sigma_b], dim=1).cpu().numpy())
                else:
                    latents.append(out[0].cpu().numpy())
        else:
            out = model.encoder(batch)
            latents.append(out.cpu().numpy())

    if _is_vae_sample:
        model.train(_was_training)

    return np.concatenate(latents, axis=0)
