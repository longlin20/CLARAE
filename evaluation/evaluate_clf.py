"""
evaluate_clf.py
===============
Unified classifier evaluation script that replaces both eval_pipeline_clf.py
and miss_classifier.py.

Combines:
  - eval_pipeline_clf.py : reconstruction metrics (R², MSE, Vpp, DF, NLEO-NRMSE),
                           denoising power, t-SNE, per-patient reconstruction,
                           clf_train.h5 / clf_test.h5 split workflow
  - miss_classifier.py   : ensemble training,
                           extra_test_split, misclassified signal HTML viewer,
                           use_ae_splits mode, DeepMLP architecture

Classifier architectures: 'mlp' (1 hidden layer) or 'deep_mlp' (residual blocks).
LogisticRegression is kept as a diagnostic baseline only.

New features:
  - Saves trained clf model to clf_model_pth/{signal_type}/{pth_stem}_clf_{clf_arch}.pth
  - Map-level classification via majority vote per map_name
  - Combined CLI flags from both source files
  - --reconstruction and --tsne flags (opt-in, False by default)
  - Returns (signal_acc, map_acc_or_None) from run_evaluate()

Usage
-----
python evaluate_clf.py \\
    --pth_path model_pth/bipolar/BIPOLAR_CLARAE_SCM_GLU_AP_ELU_GATE_SKLAST_dtw_noise_skd50.pth \\
    --data_dir processed_data_final

python evaluate_clf.py \\
    --pth_path model_pth/bipolar/*.pth \\
    --data_dir processed_data_final \\
    --clf_arch deep_mlp \\
    --wandb_project autoencoder-egms-clf
"""

import contextlib
import glob as _glob_module
import io
import json as _json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import h5py
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from run_test import build_eval_args, calculate_clinical_metrics
from run_training import set_seed
from eval_utils import (
    MLPClassifier,
    DeepMLPClassifier,
    load_model,
    extract_latent,
    _signal_type_from_stem,
    _find_dataset_path,
    _decode_str_array,
    EXCLUDED_RHYTHMS,
    RHYTHM_COLORS,
    load_split_h5,
)
from model_registry import NO_LATENT_MODELS
from preprocess_database import split_clf_dataset
from metrics.Vpp import calculate_vpp_batch
from metrics.DF import calculate_df_batch
from functions.NLEO_functions import calculateNLEORaw
from metrics.LATs_unipolar import calculate_lat_metrics_batch
from utils import add_combined_noise
from plots.plot_AE_results import visualize_test_with_noise_info
from plots.plot_misclassified import visualize_misclassified

# IAF external-validation label mapping: IAFDB rhythm name → training label name
IAF_LABEL_MAP = {'AF': 'AF', 'AFL': 'Flutter'}

# Hierarchical classifier groups
ARRHYTHMIA_LABELS = frozenset({'AF', 'Flutter'})

# ---------------------------------------------------------------------------
# Data loading — clf_train.h5 / clf_test.h5
# ---------------------------------------------------------------------------

def _decode(arr):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])


def _load_clf_h5(h5_path: str, split: str):
    """Load signals, rhythms, patient_ids, map_names, p_inf, p_sup from a clf h5 file.

    clf files (clf_train.h5, clf_test.h5) are single legacy files.
    map_names is loaded if the key '{split}_map_names' is present; otherwise None.

    Returns
    -------
    signals, rhythms, patient_ids, map_names, p_inf, p_sup
    """
    with h5py.File(h5_path, 'r') as f:
        signals     = f[f'{split}_data'][:]
        rhythms     = _decode_str_array(f[f'{split}_rhythms'][:])
        patient_ids = _decode_str_array(f[f'{split}_patient_ids'][:]) \
                      if f'{split}_patient_ids' in f \
                      else np.arange(len(rhythms)).astype(str)

        mn_key     = f'{split}_map_names'
        map_names  = _decode_str_array(f[mn_key][:]) if mn_key in f else None

        p_inf = float(f.attrs['p_inf_value']) if 'p_inf_value' in f.attrs else None
        p_sup = float(f.attrs['p_sup_value']) if 'p_sup_value' in f.attrs else None

    rhythms = np.where(rhythms == 'RS', 'SR', rhythms)
    mask    = np.array([r not in EXCLUDED_RHYTHMS for r in rhythms])
    signals     = signals[mask].astype(np.float32)
    rhythms     = rhythms[mask]
    patient_ids = patient_ids[mask]
    if map_names is not None:
        map_names = map_names[mask]

    unique, counts = np.unique(rhythms, return_counts=True)
    print(f"  [{split}] {len(rhythms):,} signals — " +
          ", ".join(f"{u}:{c:,}" for u, c in zip(unique, counts)))
    return signals, rhythms, patient_ids, map_names, p_inf, p_sup


def ensure_clf_splits(dataset_path: str, n_train_patients: int, seed: int):
    """Create clf_train.h5 / clf_test.h5 if they don't exist."""
    clf_train = os.path.join(dataset_path, 'clf_train.h5')
    clf_test  = os.path.join(dataset_path, 'clf_test.h5')
    if not (os.path.isfile(clf_train) and os.path.isfile(clf_test)):
        print("  clf splits not found — generating from test.h5 ...")
        split_clf_dataset(dataset_path, n_train_patients=n_train_patients,
                          seed=seed, output_dir=dataset_path)
    return clf_train, clf_test


def load_clf_splits(dataset_path: str,
                    clf_train_name: str = 'clf_train',
                    n_train_patients: int = 21,
                    balance_signals: bool = True,
                    seed: int = 42):
    """
    Return train and test arrays ready for MLP classification.

    If clf_train.h5 / clf_test.h5 (or the names given by clf_train_name) already
    exist in dataset_path they are reused; otherwise split_clf_dataset() is called.

    Returns
    -------
    (signals_train, rhythms_train, patient_ids_train, map_names_train),
    (signals_test,  rhythms_test,  patient_ids_test,  map_names_test),
    p_inf, p_sup
    """
    clf_test_name = clf_train_name.replace('train', 'test', 1)
    clf_train = os.path.join(dataset_path, f'{clf_train_name}.h5')
    clf_test  = os.path.join(dataset_path, f'{clf_test_name}.h5')

    if clf_train_name == 'clf_train':
        available = sorted(_glob_module.glob(os.path.join(dataset_path, 'clf_train*.h5')))
        if len(available) > 1:
            names = [os.path.basename(p) for p in available]
            print(f"\n  [INFO] Multiple clf_train*.h5 found in {dataset_path}:")
            for n in names:
                print(f"         {n}")
            print(f"  Using: {os.path.basename(clf_train)}  "
                  f"(use --clf_train_name to select another)")

    if not (os.path.isfile(clf_train) and os.path.isfile(clf_test)):
        print(f"\n{os.path.basename(clf_train)} / {os.path.basename(clf_test)} "
              f"not found — generating from test.h5 ...")
        split_clf_dataset(dataset_path,
                          n_train_patients = n_train_patients,
                          seed             = seed,
                          output_dir       = dataset_path,
                          balance_signals  = balance_signals,
                          clf_train_name   = clf_train_name)

    print(f"\nLoading clf splits from : {dataset_path}")
    print(f"  Train file : {os.path.basename(clf_train)}")
    print(f"  Test  file : {os.path.basename(clf_test)}")

    sigs_tr, rhy_tr, pids_tr, mns_tr, p_inf, p_sup = _load_clf_h5(clf_train, 'train')
    sigs_te, rhy_te, pids_te, mns_te, _,     _     = _load_clf_h5(clf_test,  'test')

    return (sigs_tr, rhy_tr, pids_tr, mns_tr), (sigs_te, rhy_te, pids_te, mns_te), p_inf, p_sup


def load_ae_splits(dataset_path: str):
    """
    Load the original AE split files (train / val / test).
    Supports both chunked ({split}_001.h5 ...) and legacy ({split}.h5) formats.

    Returns
    -------
    (signals_train, rhythms_train, patient_ids_train, map_names_train),
    (signals_val,   rhythms_val,   patient_ids_val,   map_names_val),
    (signals_test,  rhythms_test,  patient_ids_test,  map_names_test),
    p_inf, p_sup
    """
    print(f"\nLoading AE splits from : {dataset_path}")
    p_inf, p_sup = None, None
    splits_out = []
    for split in ('train', 'val', 'test'):
        d = load_split_h5(dataset_path, split,
                          include_signals=True, include_map_names=True)
        sigs  = d['signals'].astype(np.float32)
        rhy   = np.where(d['rhythms'] == 'RS', 'SR', d['rhythms'])
        pids  = d['patient_ids'] if d['patient_ids'] is not None \
                else np.arange(len(rhy)).astype(str)
        mns   = d['map_names']
        mask  = np.array([r not in EXCLUDED_RHYTHMS for r in rhy])
        sigs, rhy, pids = sigs[mask], rhy[mask], pids[mask]
        if mns is not None:
            mns = mns[mask]
        unique, counts = np.unique(rhy, return_counts=True)
        print(f"  [{split}] {len(rhy):,} signals — "
              + ", ".join(f"{u}:{c}" for u, c in zip(unique, counts)))
        splits_out.append((sigs, rhy, pids, mns))
        if p_inf is None and d['p_inf'] is not None:
            p_inf, p_sup = d['p_inf'], d['p_sup']

    return splits_out[0], splits_out[1], splits_out[2], p_inf, p_sup


# ---------------------------------------------------------------------------
# MLP helpers
# ---------------------------------------------------------------------------

def _build_clf(clf_arch: str, input_dim: int, hidden_dim: int,
               n_classes: int, dropout: float) -> nn.Module:
    if clf_arch == 'deep_mlp':
        return DeepMLPClassifier(input_dim, hidden_dim=max(hidden_dim, 256),
                                 n_classes=n_classes, dropout=dropout)
    return MLPClassifier(input_dim, hidden_dim, n_classes, dropout)


def _patient_val_split(Z, y, patient_ids, val_frac: float, seed: int):
    """Split keeping patient groups intact. val_frac of patients go to val."""
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    n_val = max(1, int(np.ceil(len(unique_patients) * val_frac)))
    val_patients = set(rng.choice(unique_patients, size=n_val, replace=False))

    tr_mask  = np.array([p not in val_patients for p in patient_ids])
    val_mask = ~tr_mask
    print(f"  Val split: {n_val}/{len(unique_patients)} patients → "
          f"{val_mask.sum():,} val / {tr_mask.sum():,} train samples")
    return (Z[tr_mask], y[tr_mask], Z[val_mask], y[val_mask])


def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def _eval(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        total_loss += criterion(logits, y).item() * len(y)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total   += len(y)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    return (total_loss / total, correct / total,
            np.concatenate(all_preds), np.concatenate(all_labels))


def train_classifier(X_train, y_train, X_test, y_test,
                     patient_ids_train=None,
                     X_val_ext=None, y_val_ext=None,
                     clf_arch='mlp',
                     hidden_dim=128, dropout=0.5,
                     lr=1e-3, weight_decay=1e-3,
                     epochs=200, batch_size=256,
                     val_frac=0.2, patience=20,
                     es_criterion='val_acc',
                     seed=42, device='cpu'):
    """Train MLP/DeepMLP with z-score normalisation and early stopping.

    If X_val_ext/y_val_ext are provided they are used directly as the validation
    set (e.g. val.h5 from the AE splits), skipping the patient-level val split.

    Returns
    -------
    clf, preds, labels, test_acc, mu, std
    """
    _all_y = [y_train, y_test]
    if y_val_ext is not None:
        _all_y.append(y_val_ext)
    n_classes = int(np.concatenate(_all_y).max()) + 1

    mu  = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0,  keepdims=True) + 1e-8
    X_train_n = (X_train - mu) / std
    X_test_n  = (X_test  - mu) / std

    if X_val_ext is not None:
        X_tr, y_tr = X_train_n, y_train
        X_val = (X_val_ext - mu) / std
        y_val = y_val_ext
        print(f"  Val set: external ({len(y_val):,} samples from val.h5)")
    elif patient_ids_train is not None:
        X_tr, y_tr, X_val, y_val = _patient_val_split(
            X_train_n, y_train, patient_ids_train, val_frac, seed)
    else:
        n_val = max(1, int(len(y_train) * val_frac))
        rng   = np.random.default_rng(seed)
        idx   = rng.permutation(len(y_train))
        X_tr, y_tr   = X_train_n[idx[n_val:]], y_train[idx[n_val:]]
        X_val, y_val = X_train_n[idx[:n_val]],  y_train[idx[:n_val]]

    clf = _build_clf(clf_arch, X_train.shape[1], hidden_dim, n_classes, dropout).to(device)
    n_params = sum(p.numel() for p in clf.parameters() if p.requires_grad)
    print(f"  Classifier: {clf_arch}  params={n_params:,}")

    optimizer = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr,  dtype=torch.float32),
                      torch.tensor(y_tr,  dtype=torch.long)),
        batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(
        TensorDataset(torch.tensor(X_test_n, dtype=torch.float32),
                      torch.tensor(y_test,   dtype=torch.long)),
        batch_size=batch_size, shuffle=False)

    use_val_loss   = es_criterion == 'val_loss'
    best_metric    = float('inf') if use_val_loss else -1.0
    best_state     = None
    best_epoch     = 0
    patience_count = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"\nTraining {clf_arch} ({epochs} epochs, patience={patience}, "
          f"criterion={es_criterion}, wd={weight_decay}) ...")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc  = _train_epoch(clf, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = _eval(clf, val_loader, criterion, device)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}"
                  f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        current  = val_loss if use_val_loss else val_acc
        improved = (current < best_metric - 1e-6) if use_val_loss \
                   else (current > best_metric + 1e-4)
        if improved:
            best_metric    = current
            best_state     = {k: v.cpu().clone() for k, v in clf.state_dict().items()}
            best_epoch     = epoch
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best {es_criterion}={best_metric:.4f} at epoch {best_epoch})")
                break

    if best_state is not None:
        clf.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"  Restored best model from epoch {best_epoch} "
              f"({es_criterion}={best_metric:.4f})")

    _, test_acc, preds, labels = _eval(clf, test_loader, criterion, device)
    print(f"\nTest accuracy: {test_acc:.4f}")
    return clf, preds, labels, test_acc, mu, std, history


@torch.no_grad()
def _clf_predict(clf, X_norm: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    """Run MLP inference on already z-score-normalised latent vectors."""
    clf.eval()
    preds = []
    ds = DataLoader(TensorDataset(torch.tensor(X_norm, dtype=torch.float32)),
                    batch_size=batch_size, shuffle=False)
    for (batch,) in ds:
        preds.append(clf(batch.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def train_ensemble(X_train, y_train, X_test, y_test,
                   patient_ids_train=None,
                   n_members=5,
                   clf_arch='mlp',
                   hidden_dim=128, dropout=0.5,
                   lr=1e-3, weight_decay=1e-3,
                   epochs=200, batch_size=256,
                   val_frac=0.2, patience=20,
                   es_criterion='val_acc',
                   seed=42, device='cpu'):
    """Train N MLPs with different val-patient splits. Average softmax → final prediction."""
    import torch.nn.functional as F

    mu  = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0,  keepdims=True) + 1e-8
    X_tr_s = (X_train - mu) / std
    X_te_s = (X_test  - mu) / std

    n_classes   = len(np.unique(y_train))
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_te_s, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False)

    all_probs = []
    for m in range(n_members):
        print(f"\n--- Ensemble member {m+1}/{n_members} (seed={seed+m}) ---")
        clf_m, _, _, _, _, _, _ = train_classifier(
            X_train, y_train, X_test, y_test,
            patient_ids_train = patient_ids_train,
            clf_arch          = clf_arch,
            hidden_dim        = hidden_dim,
            dropout           = dropout,
            lr                = lr,
            weight_decay      = weight_decay,
            epochs            = epochs,
            batch_size        = batch_size,
            val_frac          = val_frac,
            patience          = patience,
            es_criterion      = es_criterion,
            seed              = seed + m,
            device            = device,
        )
        clf_m.eval()
        probs_m = []
        with torch.no_grad():
            for (X,) in test_loader:
                probs_m.append(F.softmax(clf_m(X.to(device)), dim=1).cpu().numpy())
        all_probs.append(np.concatenate(probs_m, axis=0))

    avg_probs = np.mean(all_probs, axis=0)
    preds     = avg_probs.argmax(axis=1)
    labels    = y_test
    test_acc  = float(accuracy_score(labels, preds))
    print(f"\nEnsemble ({n_members} members) test_acc: {test_acc:.4f}")
    return preds, labels, test_acc, mu, std


def train_logreg(X_train, y_train, X_test, y_test, C=0.01, seed=42):
    """Logistic Regression baseline (sklearn) — diagnostic only."""
    mu  = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0,  keepdims=True) + 1e-8
    X_tr_s = (X_train - mu) / std
    X_te_s = (X_test  - mu) / std

    print(f"\nTraining LogisticRegression (C={C}) ...")
    lr_clf = LogisticRegression(C=C, max_iter=1000, solver='lbfgs',
                                random_state=seed, n_jobs=-1)
    lr_clf.fit(X_tr_s, y_train)
    preds    = lr_clf.predict(X_te_s)
    test_acc = float(accuracy_score(y_test, preds))
    tr_acc   = float(accuracy_score(y_train, lr_clf.predict(X_tr_s)))
    print(f"  LogReg train_acc={tr_acc:.4f}  test_acc={test_acc:.4f}")
    return preds, test_acc, lr_clf, mu, std


# ---------------------------------------------------------------------------
# Save classifier model
# ---------------------------------------------------------------------------

def _save_clf_model(clf: nn.Module, clf_arch: str, class_names,
                    X_train_shape1: int, mlp_hidden_dim: int,
                    signal_type: str, pth_stem: str,
                    mu: np.ndarray, std: np.ndarray,
                    save_dir: str = None):
    """Save the trained classifier state dict.

    save_dir : explicit output directory. If None, defaults to
               clf_model_pth/{signal_type}/.
    """
    if save_dir is None:
        save_dir = os.path.join('clf_model_pth', signal_type)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{pth_stem}_clf_{clf_arch}.pth')
    torch.save({
        'state_dict':  clf.state_dict(),
        'class_names': list(class_names),
        'input_dim':   X_train_shape1,
        'hidden_dim':  mlp_hidden_dim,
        'n_classes':   len(class_names),
        'clf_arch':    clf_arch,
        'mu':          mu,
        'std':         std,
    }, save_path)
    print(f"  Classifier saved → {save_path}")
    return save_path


def _load_clf_model(clf_pth: str, device):
    """Load a saved classifier from clf_model_pth/.

    Returns
    -------
    clf, class_names, mu, std, clf_arch
    """
    ckpt = torch.load(clf_pth, map_location=device, weights_only=False)
    clf_arch    = ckpt['clf_arch']
    class_names = np.array(ckpt['class_names'])
    mu  = ckpt['mu']
    std = ckpt['std']
    clf = _build_clf(clf_arch, ckpt['input_dim'], ckpt['hidden_dim'],
                     ckpt['n_classes'], dropout=0.0)
    clf.load_state_dict(ckpt['state_dict'])
    clf = clf.to(device)
    clf.eval()
    print(f"  Classifier loaded  ← {clf_pth}")
    print(f"  arch={clf_arch}  classes={list(class_names)}")
    return clf, class_names, mu, std, clf_arch


# ---------------------------------------------------------------------------
# Map-level classification
# ---------------------------------------------------------------------------

def _majority(votes: list) -> int:
    counts = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: counts[k])


def _clf_per_map(preds, labels, map_names, class_names,
                 signals_per_pred: int = 0):
    """
    Group signal-level predictions by map_name and compute majority-vote prediction.

    Parameters
    ----------
    preds            : (N,) int array of predicted class indices
    labels           : (N,) int array of true class indices
    map_names        : (N,) str array of map identifiers
    class_names      : sequence of class name strings
    signals_per_pred : if > 0, split each map's signals into non-overlapping
                       blocks of this size; each block casts one vote (majority
                       of the block's signal predictions); leftover signals
                       that don't fill a complete block are discarded.
                       Maps with fewer signals than signals_per_pred are skipped.
                       If 0 (default), all signals vote directly.

    Returns
    -------
    map_preds        : (M,) int array — majority-vote prediction per map
    map_labels       : (M,) int array — true label per map
    map_names_unique : (M,) str array — unique map identifiers (in order of
                       first occurrence); may be shorter than input if maps
                       are skipped due to insufficient signals.
    """
    seen = {}
    order = []
    for i, mn in enumerate(map_names):
        if mn not in seen:
            seen[mn] = {'preds': [], 'label': int(labels[i])}
            order.append(mn)
        seen[mn]['preds'].append(int(preds[i]))

    map_preds_list  = []
    map_labels_list = []
    order_out       = []

    for mn in order:
        entry     = seen[mn]
        sig_preds = entry['preds']

        if signals_per_pred and signals_per_pred > 0:
            n_blocks = len(sig_preds) // signals_per_pred
            if n_blocks == 0:          # not enough signals → skip map
                continue
            for b in range(n_blocks):
                block = sig_preds[b * signals_per_pred:(b + 1) * signals_per_pred]
                map_preds_list.append(_majority(block))
                map_labels_list.append(entry['label'])
                order_out.append(mn)
        else:
            map_preds_list.append(_majority(sig_preds))
            map_labels_list.append(entry['label'])
            order_out.append(mn)

    return (np.array(map_preds_list, dtype=int),
            np.array(map_labels_list, dtype=int),
            np.array(order_out))


# ---------------------------------------------------------------------------
# Reconstruction helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _reconstruct(ae_model, signals: np.ndarray, batch_size: int, device):
    """Return reconstructed signals (N, 1, L) as numpy float32."""
    ae_model.eval()
    recons = []
    tensor = torch.from_numpy(signals)
    for (batch,) in DataLoader(TensorDataset(tensor), batch_size=batch_size):
        recons.append(ae_model(batch.to(device)).cpu().numpy())
    return np.concatenate(recons, axis=0)


def _r2_per_patient(signals, recons, patient_ids):
    """Return dict {patient_id: r2}."""
    out = {}
    for pid in np.unique(patient_ids):
        mask   = patient_ids == pid
        y_true = signals[mask].reshape(-1)
        y_pred = recons[mask].reshape(-1)
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        out[pid] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return out


def _r2_signal_stats(signals: np.ndarray, recons: np.ndarray):
    """Per-signal R²: mean, std, min, max."""
    r2s = []
    for i in range(len(signals)):
        y_t    = signals[i].reshape(-1)
        y_p    = recons[i].reshape(-1)
        ss_res = ((y_t - y_p) ** 2).sum()
        ss_tot = ((y_t - y_t.mean()) ** 2).sum()
        r2s.append(float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0)
    r2s = np.array(r2s)
    return float(np.mean(r2s)), float(np.std(r2s)), float(np.min(r2s)), float(np.max(r2s))


def _clinical_metrics(signals, recons, patient_ids, is_bipolar: bool, fs: int = 500,
                      p_inf=None, p_sup=None):
    """
    Compute R², Vpp error, DF error, LAT (unipolar) or NLEO corr (bipolar)
    per signal, looping patient by patient to avoid OOM.

    Large intermediate arrays (denormalized, flat) are created only for one
    patient's signals at a time and freed after each iteration. Per-signal
    scalar values are accumulated in lists (negligible memory).

    Returns
    -------
    patient_rows : list of dicts, one per patient (nanmean of signal metrics)
    summary      : dict with mean ± std across ALL signals
    """
    orig_2d  = signals.squeeze(1)
    recon_2d = recons.squeeze(1)
    n_total  = len(orig_2d)
    scale    = (p_sup - p_inf) if (p_inf is not None and p_sup is not None) else None

    pids_unique = sorted(np.unique(patient_ids))
    n_pids      = len(pids_unique)

    # Accumulators — just floats, negligible memory
    all_r2   = []
    all_vpp  = []
    all_df   = []
    all_lat_mae      = [] if not is_bipolar else None
    all_lat_unm      = [] if not is_bipolar else None
    all_lat_unm_recon = [] if not is_bipolar else None
    all_nleo    = [] if is_bipolar  else None

    patient_rows = []

    for idx, pid in enumerate(pids_unique):
        mask = patient_ids == pid
        o    = orig_2d[mask]    # (n_pat, L) — small
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

        # ── LAT per signal (unipolar) — one batch call per patient ───────────
        if not is_bipolar:
            print(f"{prefix} — LAT ...", flush=True)
            lat_mae_p      = np.full(len(o), float('nan'))
            lat_unm_p      = np.full(len(o), float('nan'))
            lat_unm_recon_p = np.full(len(o), float('nan'))
            try:
                lat_res         = calculate_lat_metrics_batch(o_ds, r_ds, fs=fs)
                lat_mae_p       = lat_res['LAT_matched_MAE_ms']['values'].astype(float)
                lat_unm_p       = lat_res['LAT_unmatched_orig']['values'].astype(float)
                lat_unm_recon_p = lat_res['LAT_unmatched_recon']['values'].astype(float)
            except Exception:
                pass
            row['lat_mae_ms']         = float(np.nanmean(lat_mae_p))
            row['lat_unmatched_orig']  = float(np.nanmean(lat_unm_p))
            row['lat_unmatched_recon'] = float(np.nanmean(lat_unm_recon_p))
            all_lat_mae.extend(lat_mae_p.tolist())
            all_lat_unm.extend(lat_unm_p.tolist())
            all_lat_unm_recon.extend(lat_unm_recon_p.tolist())

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

        # Accumulate per-signal scalars (tiny memory)
        all_r2.extend(r2_p.tolist())
        all_vpp.extend(vpp_p.tolist())
        all_df.extend(df_p.tolist())

        patient_rows.append(row)

    # ── Summary: mean ± std across ALL signals ────────────────────────────────
    all_r2  = np.array(all_r2);  all_vpp = np.array(all_vpp); all_df = np.array(all_df)
    summary = {
        'r2_mean':        float(np.nanmean(all_r2)),
        'r2_std':         float(np.nanstd(all_r2)),
        'vpp_error_mean': float(np.nanmean(all_vpp)),
        'vpp_error_std':  float(np.nanstd(all_vpp)),
        'df_error_mean':  float(np.nanmean(all_df)),
        'df_error_std':   float(np.nanstd(all_df)),
    }
    if not is_bipolar:
        a = np.array(all_lat_mae); b = np.array(all_lat_unm); c = np.array(all_lat_unm_recon)
        summary['lat_mae_ms_mean']          = float(np.nanmean(a))
        summary['lat_mae_ms_std']           = float(np.nanstd(a))
        summary['lat_unmatched_orig_mean']  = float(np.nanmean(b))
        summary['lat_unmatched_orig_std']   = float(np.nanstd(b))
        summary['lat_unmatched_recon_mean'] = float(np.nanmean(c))
        summary['lat_unmatched_recon_std']  = float(np.nanstd(c))
    if is_bipolar:
        a = np.array(all_nleo)
        summary['nleo_corr_mean'] = float(np.nanmean(a))
        summary['nleo_corr_std']  = float(np.nanstd(a))

    print(f"\n  Total {n_total} señales — "
          f"r2={summary['r2_mean']:.4f}±{summary['r2_std']:.4f}  "
          f"vpp={summary['vpp_error_mean']:.4f}±{summary['vpp_error_std']:.4f}  "
          f"df={summary['df_error_mean']:.4f}±{summary['df_error_std']:.4f}", flush=True)

    return patient_rows, summary


# ---------------------------------------------------------------------------
# Per-patient classification metrics
# ---------------------------------------------------------------------------

def _clf_per_patient(preds, labels, patient_ids, class_names):
    """Return list of dicts with per-patient accuracy, macro-f1, and per-class f1."""
    rows = []
    for pid in sorted(np.unique(patient_ids)):
        mask = patient_ids == pid
        p, l = preds[mask], labels[mask]
        row  = {
            'patient':   pid,
            'n_signals': int(mask.sum()),
            'accuracy':  float(accuracy_score(l, p)),
            'macro_f1':  float(f1_score(l, p, average='macro', zero_division=0)),
        }
        f1_cls = f1_score(l, p, average=None,
                          labels=list(range(len(class_names))),
                          zero_division=0)
        for cls, f1v in zip(class_names, f1_cls):
            row[f'f1_{cls}'] = float(f1v)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Error breakdown: which patient / map failed
# ---------------------------------------------------------------------------

def _error_by_patient_map(preds, labels, patient_ids, map_names, class_names,
                           failed_maps=None, map_pred_override=None):
    """
    One row per failed map: shows what the map was predicted as (from map-level
    voting), plus signal-level error count and rate.

    Parameters
    ----------
    failed_maps       : set or None — only include maps in this set.
    map_pred_override : dict {map_name: predicted_class_str} or None.
        When provided, 'predicted' column uses the actual map/block-level
        prediction (consistent with the confusion matrix) instead of the
        most common signal-level error.

    Returns list of dicts:
      map | true | predicted | n_errors | n_total | error_rate
    """
    from collections import defaultdict

    map_data = {}
    for i in range(len(labels)):
        mn = map_names[i] if map_names is not None else 'N/A'
        if failed_maps is not None and mn not in failed_maps:
            continue
        if mn not in map_data:
            map_data[mn] = {'total': 0, 'patient': patient_ids[i],
                            'true': class_names[labels[i]],
                            'err_counts': defaultdict(int)}
        map_data[mn]['total'] += 1
        if preds[i] != labels[i]:
            map_data[mn]['err_counts'][class_names[preds[i]]] += 1

    rows = []
    for mn, data in map_data.items():
        if not data['err_counts']:
            continue
        if map_pred_override and mn in map_pred_override:
            predicted = map_pred_override[mn]
        else:
            predicted = max(data['err_counts'], key=lambda k: data['err_counts'][k])
        n_err = sum(data['err_counts'].values())
        n_tot = data['total']
        rows.append({
            'patient':    data['patient'],
            'map':        mn,
            'true':       data['true'],
            'predicted':  predicted,
            'n_errors':   n_err,
            'n_total':    n_tot,
            'error_rate': round(n_err / n_tot, 4),
        })

    rows.sort(key=lambda r: r['n_errors'], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# WandB table helper
# ---------------------------------------------------------------------------

def _log_wandb_table(rows, key):
    if not (WANDB_AVAILABLE and wandb.run is not None) or not rows:
        return
    columns = list(rows[0].keys())
    data    = [[row[c] for c in columns] for row in rows]
    wandb.log({key: wandb.Table(columns=columns, data=data)})





# ---------------------------------------------------------------------------
# _run_one: manages wandb init/finish for one pth
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5-fold CV helpers
# ---------------------------------------------------------------------------

def _make_cv_folds(pids_unique, rhy_all, pids_all,
                   n_folds=5, seed=42, max_attempts=5000):
    """
    Patient-level k-fold split.
    Fold sizes: [ceil(N/k)] × (N%k), then [floor(N/k)] × remaining.
    Constraint: each test fold must contain ≥1 signal of every rhythm class.
    Returns list of (test_pids_set, train_pids_set) per fold.
    """
    rng = np.random.default_rng(seed)
    N = len(pids_unique)
    base, extra = divmod(N, n_folds)
    fold_sizes = [base + 1] * extra + [base] * (n_folds - extra)

    pid_to_rhythms = {}
    for pid, rhy in zip(pids_all, rhy_all):
        pid_to_rhythms.setdefault(pid, set()).add(rhy)
    all_rhythms = set(rhy_all)
    all_pids_set = set(pids_unique)

    for attempt in range(max_attempts):
        perm = rng.permutation(N)
        folds_test, start = [], 0
        for sz in fold_sizes:
            folds_test.append(set(pids_unique[perm[start:start + sz]]))
            start += sz

        valid = all(
            all_rhythms.issubset(
                {r for pid in ft for r in pid_to_rhythms.get(pid, set())}
            )
            for ft in folds_test
        )
        if valid:
            print(f"  [CV] Valid split found at attempt {attempt + 1}  "
                  f"fold sizes: {fold_sizes}")
            return [(ft, all_pids_set - ft) for ft in folds_test]

    raise RuntimeError(
        f"No valid CV split found after {max_attempts} attempts. "
        f"Check rhythm distribution across patients."
    )


def run_cv_evaluate(pth_path, data_dir=None, dataset_path=None,
                    n_folds=5, seed=42,
                    latent_dim=64, filters_initial=64, dense_dim=128,
                    q_parameter=2, dropout_rate=0.1, loss_function=None,
                    clf_arch='mlp', mlp_hidden_dim=128, mlp_lr=1e-3,
                    mlp_epochs=200, mlp_dropout=0.5, mlp_weight_decay=1e-3,
                    mlp_val_frac=0.2, mlp_patience=20,
                    mlp_es_criterion='val_acc',
                    map_signals_per_pred=0,
                    use_logreg=False,
                    batch_size=256, device='cpu',
                    noise_snr_min: float = -5.0,
                    noise_snr_max: float = 10.0,
                    sampling_freq: int = 500,
                    no_clf: bool = False,
                    no_recon: bool = False,
                    viz_args=None,
                    viz_only: bool = False,
                    do_tsne: bool = False,
                    iaf_h5: str = None,
                    hierarchical: bool = False,
                    raw_signals: bool = False):
    """
    Evaluate a single .pth on test.h5.

    When no_clf=False (default): 5-fold CV classification + recon/denoise tables.
    When no_clf=True: skip encoding and classification, only compute per-patient
    recon/denoise tables (faster, no MLP training).
    Logs to the active WandB run (caller must init/finish).
    """
    device_obj = torch.device(device)
    pth_stem   = os.path.splitext(os.path.basename(pth_path))[0]

    # ── Resolve architecture ────────────────────────────────────────────────
    try:
        eval_args = build_eval_args(
            pth_stem,
            argparse.Namespace(loss_function=loss_function,
                               device=str(device_obj),
                               wandb_project=None, wandb_entity=None,
                               no_wandb=True),
        )
    except KeyError as e:
        raise RuntimeError(f"build_eval_args failed for {pth_stem}: {e}")

    eval_args.latent_dim      = latent_dim
    eval_args.filters_initial = filters_initial
    eval_args.dense_dim       = dense_dim

    signal_type = _signal_type_from_stem(pth_stem)
    if dataset_path is None:
        if data_dir is None:
            raise ValueError("Provide data_dir or dataset_path.")
        dataset_path = _find_dataset_path(data_dir, signal_type)
    print(f"  Dataset : {dataset_path}")
    print(f"  Arch    : {eval_args.model_architecture}")

    has_latent = eval_args.model_architecture not in NO_LATENT_MODELS
    if not has_latent and not no_clf and not raw_signals:
        print(f"[INFO] {eval_args.model_architecture}: no latent space — CV skipped.")
        return {}
    if not has_latent and no_clf:
        print(f"[INFO] {eval_args.model_architecture}: no latent space — running recon/denoise only.")

    # ── Load test split (chunked test_001.h5 … or legacy test.h5) ───────────
    print(f"\n[CV] Loading test split from {dataset_path}")
    d = load_split_h5(dataset_path, 'test', include_signals=True, include_map_names=True)
    sigs_all = d['signals'].astype(np.float32)
    rhy_all  = np.where(d['rhythms'] == 'RS', 'SR', d['rhythms'])
    pids_all = d['patient_ids'] if d['patient_ids'] is not None \
               else np.arange(len(rhy_all)).astype(str)
    mns_all  = d['map_names']
    p_inf = d.get('p_inf'); p_sup = d.get('p_sup')
    is_bipolar = not eval_args.unipolar
    mask     = np.array([r not in EXCLUDED_RHYTHMS for r in rhy_all])
    sigs_all, rhy_all, pids_all = sigs_all[mask], rhy_all[mask], pids_all[mask]
    if mns_all is not None:
        mns_all = mns_all[mask]
    unique, counts = np.unique(rhy_all, return_counts=True)
    print(f"  [test] {len(rhy_all):,} signals — "
          + ", ".join(f"{u}:{c:,}" for u, c in zip(unique, counts)))

    # ── Load AE and encode all signals once ─────────────────────────────────
    if raw_signals and no_recon:
        ae_model = None
        print(f"\n[raw_signals] Skipping AE — using flattened raw signals as features.")
    else:
        print(f"\n[CV] Loading AE: {pth_path}")
        ae_model = load_model(
            pth_path        = pth_path,
            model_class     = eval_args.model_architecture,
            latent_dim      = eval_args.latent_dim,
            filters_initial = eval_args.filters_initial,
            dropout_rate    = eval_args.dropout_rate,
            dense_dim       = eval_args.dense_dim,
            input_length    = sigs_all.shape[2],
            device          = str(device_obj),
            q_parameter     = eval_args.q_parameter,
        )
    # ── viz_only: skip all metrics, just generate the reconstruction plots ────
    if viz_only and viz_args is not None:
        print(f"\n[viz-only] Generating reconstruction visualizations ({viz_args.max_signals} samples) ...")
        _data_t = torch.from_numpy(sigs_all).float()
        _p_inf_viz = p_inf if p_inf is not None else 0.0
        _p_sup_viz = p_sup if p_sup is not None else 1.0
        visualize_test_with_noise_info(
            ae_model, _data_t, _p_inf_viz, _p_sup_viz,
            label='test', num_samples=viz_args.max_signals,
            args=viz_args, device=str(device_obj),
            is_bipolar=is_bipolar,
        )
        return {}

    if no_clf and not do_tsne:
        print(f"\n[Recon-only] Skipping encoding and classification.")
    elif raw_signals:
        n_features = sigs_all.shape[1] * sigs_all.shape[2]
        print(f"\n[raw_signals] Flattening {len(sigs_all):,} signals → {n_features}D features ...")
        X_all = sigs_all.reshape(len(sigs_all), -1)
        print(f"  X_all: {X_all.shape}")
    else:
        print(f"\n[CV] Encoding {len(sigs_all):,} signals ...")
        X_all = extract_latent(ae_model, sigs_all, eval_args.model_architecture,
                               batch_size, str(device_obj))
        print(f"  X_all: {X_all.shape}")

    if not no_clf:
        # ── Labels ──────────────────────────────────────────────────────────────
        le = LabelEncoder()
        le.fit(rhy_all)
        class_names = le.classes_
        y_all = le.transform(rhy_all)
        print(f"  Classes: {list(class_names)}")

        # ── IAF external test data (optional) ───────────────────────────────────
        X_iaf = None; y_iaf_enc = None; pids_iaf = None; iaf_class_present = []; rhy_iaf_orig = None
        if iaf_h5 is not None:
            print(f"\n[IAF] Loading {iaf_h5} ...")
            with h5py.File(iaf_h5, 'r') as _f:
                _sigs_iaf = _f['signals'][:].astype(np.float32)
                _rhy_iaf  = np.array([r.decode() if isinstance(r, bytes) else r
                                       for r in _f['rhythms'][:]])
                _pids_iaf = np.array([p.decode() if isinstance(p, bytes) else p
                                       for p in _f['patient_ids'][:]])
            _rhy_mapped = np.array([IAF_LABEL_MAP.get(r, r) for r in _rhy_iaf])
            _valid      = np.array([r in le.classes_ for r in _rhy_mapped])
            if _valid.sum() == 0:
                print(f"[IAF] No IAF labels match training classes {list(le.classes_)}. Skipping.")
            else:
                _sigs_iaf   = _sigs_iaf[_valid]
                _rhy_mapped = _rhy_mapped[_valid]
                pids_iaf    = _pids_iaf[_valid]
                _u, _c = np.unique(_rhy_mapped, return_counts=True)
                print(f"  {_valid.sum():,} IAF signals — "
                      + ", ".join(f"{u_}:{c_:,}" for u_, c_ in zip(_u, _c)))
                if raw_signals:
                    print(f"[IAF] Flattening raw IAF signals → {_sigs_iaf.shape[1]*_sigs_iaf.shape[2]}D ...")
                    X_iaf = _sigs_iaf.reshape(len(_sigs_iaf), -1)
                else:
                    print(f"[IAF] Encoding IAF signals with AE ...")
                    X_iaf = extract_latent(ae_model, _sigs_iaf,
                                           eval_args.model_architecture,
                                           batch_size, str(device_obj))
                y_iaf_enc         = le.transform(_rhy_mapped)
                iaf_class_present = sorted(set(_rhy_mapped))
                rhy_iaf_orig      = _rhy_mapped.copy()   # rhythm strings, for hier stage 2
                print(f"  X_iaf: {X_iaf.shape}  classes: {iaf_class_present}")

        # ── Build folds ─────────────────────────────────────────────────────────
        pids_unique = np.unique(pids_all)
        print(f"\n[CV] {len(pids_unique)} unique patients → {n_folds} folds  (seed={seed}, fixed for all models)")
        folds = _make_cv_folds(pids_unique, rhy_all, pids_all,
                               n_folds=n_folds, seed=seed)

        # Print and log fold patient assignments — identical for every model using the same dataset+seed
        fold_assign_rows = []
        print(f"[CV] Fold assignments (deterministic: same dataset + seed={seed} → same folds):")
        for fold_idx, (test_pids, _) in enumerate(folds):
            sorted_pids = sorted(test_pids)
            print(f"  Fold {fold_idx+1}: test = {sorted_pids}")
            for pid in sorted_pids:
                fold_assign_rows.append({'fold': fold_idx + 1, 'patient': pid})

        # ── Per-fold training and evaluation ────────────────────────────────────
        fold_accs          = []
        fold_macro_f1      = []
        fold_perclass      = {cls: [] for cls in class_names}
        fold_map_accs      = []
        fold_map_macro_f1  = []
        all_patient_rows   = []   # per-patient rows across all folds
        # Aggregated preds/labels across all folds (each sample in exactly one fold)
        agg_preds          = []
        agg_labels         = []
        agg_map_preds      = []
        agg_map_labels     = []
        # LR per-fold tracking
        fold_lr_accs       = []
        fold_lr_macro_f1   = []
        fold_lr_perclass   = {cls: [] for cls in class_names}
        agg_lr_preds       = []
        # IAF per-fold tracking
        fold_iaf_accs      = []
        fold_iaf_macro_f1  = []
        fold_iaf_perclass  = {cls: [] for cls in iaf_class_present}
        fold_iaf_pat_accs  = []

        for fold_idx, (test_pids, train_pids) in enumerate(folds):
            print(f"\n{'─'*60}")
            n_te_pids = len(test_pids)
            n_tr_pids = len(train_pids)
            print(f"  [CV Fold {fold_idx+1}/{n_folds}]  "
                  f"test={n_te_pids} patients  train={n_tr_pids} patients")

            te_mask = np.isin(pids_all, list(test_pids))
            tr_mask = ~te_mask

            X_tr  = X_all[tr_mask];  y_tr  = y_all[tr_mask]
            X_te  = X_all[te_mask];  y_te  = y_all[te_mask]
            pids_tr = pids_all[tr_mask]
            pids_te = pids_all[te_mask]
            mns_te  = mns_all[te_mask] if mns_all is not None else None

            clf_fold, preds_fold, labels_fold, acc_fold, mu_fold, std_fold, _ = train_classifier(
                X_tr, y_tr, X_te, y_te,
                patient_ids_train = pids_tr,
                clf_arch          = clf_arch,
                hidden_dim        = mlp_hidden_dim,
                dropout           = mlp_dropout,
                lr                = mlp_lr,
                weight_decay      = mlp_weight_decay,
                epochs            = mlp_epochs,
                batch_size        = batch_size,
                val_frac          = mlp_val_frac,
                patience          = mlp_patience,
                es_criterion      = mlp_es_criterion,
                seed              = seed + fold_idx,
                device            = str(device_obj),
            )

            report = classification_report(
                labels_fold, preds_fold,
                target_names=class_names,
                output_dict=True, digits=4, zero_division=0,
            )
            macro = float(report['macro avg']['f1-score'])
            fold_accs.append(float(acc_fold))
            fold_macro_f1.append(macro)
            for cls in class_names:
                fold_perclass[cls].append(float(report.get(cls, {}).get('f1-score', 0.0)))

            agg_preds.extend(preds_fold.tolist())
            agg_labels.extend(labels_fold.tolist())

            # LR baseline per fold
            if use_logreg:
                lr_preds_fold, lr_acc_fold, _, _, _ = train_logreg(
                    X_tr, y_tr, X_te, y_te, seed=seed + fold_idx)
                lr_report = classification_report(
                    labels_fold, lr_preds_fold,
                    target_names=class_names,
                    output_dict=True, digits=4, zero_division=0,
                )
                lr_macro = float(lr_report['macro avg']['f1-score'])
                fold_lr_accs.append(float(lr_acc_fold))
                fold_lr_macro_f1.append(lr_macro)
                for cls in class_names:
                    fold_lr_perclass[cls].append(
                        float(lr_report.get(cls, {}).get('f1-score', 0.0)))
                agg_lr_preds.extend(lr_preds_fold.tolist())
                print(f"  Fold {fold_idx+1} LR : acc={lr_acc_fold:.4f}  macro_f1={lr_macro:.4f}")

            # Per-patient metrics for this fold (each patient appears in exactly one test fold)
            patient_rows_fold = _clf_per_patient(preds_fold, labels_fold, pids_te, class_names)
            for row in patient_rows_fold:
                row['fold'] = fold_idx + 1
            all_patient_rows.extend(patient_rows_fold)

            print(f"  Fold {fold_idx+1}: acc={acc_fold:.4f}  macro_f1={macro:.4f}")
            print(f"  Per-patient ({len(patient_rows_fold)} patients):")
            for row in patient_rows_fold:
                print(f"    {row['patient']:<15} n={row['n_signals']:>4}  "
                      f"acc={row['accuracy']:.4f}  macro_f1={row['macro_f1']:.4f}")

            # Map-level majority vote for this fold
            if mns_te is not None:
                map_preds_fold, map_labels_fold, _ = _clf_per_map(
                    preds_fold, labels_fold, mns_te, class_names,
                    signals_per_pred=map_signals_per_pred)
                map_acc_fold  = float(accuracy_score(map_labels_fold, map_preds_fold))
                map_mf1_fold  = float(f1_score(map_labels_fold, map_preds_fold,
                                               average='macro', zero_division=0))
                fold_map_accs.append(map_acc_fold)
                fold_map_macro_f1.append(map_mf1_fold)
                agg_map_preds.extend(map_preds_fold.tolist())
                agg_map_labels.extend(map_labels_fold.tolist())
                print(f"  Fold {fold_idx+1} map: acc={map_acc_fold:.4f}  macro_f1={map_mf1_fold:.4f}")

            # ── IAF inference for this fold ──────────────────────────────────────
            if X_iaf is not None:
                X_iaf_n   = (X_iaf - mu_fold) / std_fold
                preds_iaf = _clf_predict(clf_fold, X_iaf_n, str(device_obj), batch_size)
                acc_iaf   = float(accuracy_score(y_iaf_enc, preds_iaf))
                rep_iaf   = classification_report(
                    y_iaf_enc, preds_iaf,
                    target_names=class_names, output_dict=True, digits=4, zero_division=0)
                macro_iaf = float(rep_iaf['macro avg']['f1-score'])
                fold_iaf_accs.append(acc_iaf)
                fold_iaf_macro_f1.append(macro_iaf)
                for cls in iaf_class_present:
                    fold_iaf_perclass[cls].append(
                        float(rep_iaf.get(cls, {}).get('f1-score', 0.0)))
                # Patient-level majority vote on IAF patients
                iaf_pat_preds, iaf_pat_labels, iaf_pat_names = _clf_per_map(
                    preds_iaf, y_iaf_enc, pids_iaf, class_names)
                iaf_pat_acc = float(accuracy_score(iaf_pat_labels, iaf_pat_preds))
                fold_iaf_pat_accs.append(iaf_pat_acc)
                print(f"  Fold {fold_idx+1} IAF: signal_acc={acc_iaf:.4f}  "
                      f"macro_f1={macro_iaf:.4f}  patient_acc={iaf_pat_acc:.4f}")
                for cls in iaf_class_present:
                    recall = float(rep_iaf.get(cls, {}).get('recall', 0.0))
                    print(f"    IAF {cls}: recall={recall:.4f}  "
                          f"f1={float(rep_iaf.get(cls, {}).get('f1-score', 0.0)):.4f}")
                # Per-fold confusion matrix
                _sz_iaf = (max(5, len(class_names) + 1), max(4, len(class_names)))
                fig_cm_iaf, ax_iaf = plt.subplots(figsize=_sz_iaf)
                ConfusionMatrixDisplay(
                    confusion_matrix=confusion_matrix(y_iaf_enc, preds_iaf),
                    display_labels=class_names).plot(
                    ax=ax_iaf, colorbar=True, cmap='Oranges', xticks_rotation=45)
                ax_iaf.set_title(f"IAF Fold {fold_idx+1} – {pth_stem}\n"
                                 f"signal_acc={acc_iaf:.4f}  patient_acc={iaf_pat_acc:.4f}")
                plt.tight_layout()
                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({f'iaf/confusion_matrix_fold{fold_idx+1}': wandb.Image(fig_cm_iaf)})
                plt.close(fig_cm_iaf)

        # ── Aggregate ───────────────────────────────────────────────────────────
        acc_mean  = float(np.mean(fold_accs));    acc_std  = float(np.std(fold_accs))
        mf1_mean  = float(np.mean(fold_macro_f1)); mf1_std = float(np.std(fold_macro_f1))

        print(f"\n{'='*60}")
        print(f"  [CV] accuracy  : {acc_mean:.4f} ± {acc_std:.4f}  "
              f"(folds: {[f'{v:.4f}' for v in fold_accs]})")
        print(f"  [CV] macro_f1  : {mf1_mean:.4f} ± {mf1_std:.4f}  "
              f"(folds: {[f'{v:.4f}' for v in fold_macro_f1]})")
        for cls in class_names:
            m = float(np.mean(fold_perclass[cls]))
            s = float(np.std(fold_perclass[cls]))
            print(f"  [CV] {cls:<12}: f1 = {m:.4f} ± {s:.4f}")

        log_dict = {
            'clf/test_acc_mean':  acc_mean,  'clf/test_acc_std':  acc_std,
            'clf/macro_f1_mean':  mf1_mean,  'clf/macro_f1_std':  mf1_std,
            'clf/n_folds':        n_folds,
            'clf/n_patients':     int(len(pids_unique)),
        }
        for cls in class_names:
            log_dict[f'clf/{cls}_f1-score_mean'] = float(np.mean(fold_perclass[cls]))
            log_dict[f'clf/{cls}_f1-score_std']  = float(np.std(fold_perclass[cls]))

        # ── IAF aggregate ────────────────────────────────────────────────────────
        if fold_iaf_accs:
            iaf_acc_mean = float(np.mean(fold_iaf_accs))
            iaf_acc_std  = float(np.std(fold_iaf_accs))
            iaf_mf1_mean = float(np.mean(fold_iaf_macro_f1))
            iaf_mf1_std  = float(np.std(fold_iaf_macro_f1))
            iaf_pat_mean = float(np.mean(fold_iaf_pat_accs))
            iaf_pat_std  = float(np.std(fold_iaf_pat_accs))
            print(f"\n  [IAF] signal_acc : {iaf_acc_mean:.4f} ± {iaf_acc_std:.4f}  "
                  f"(folds: {[f'{v:.4f}' for v in fold_iaf_accs]})")
            print(f"  [IAF] macro_f1   : {iaf_mf1_mean:.4f} ± {iaf_mf1_std:.4f}")
            print(f"  [IAF] patient_acc: {iaf_pat_mean:.4f} ± {iaf_pat_std:.4f}")
            for cls in iaf_class_present:
                m = float(np.mean(fold_iaf_perclass[cls]))
                s = float(np.std(fold_iaf_perclass[cls]))
                print(f"  [IAF] {cls:<12}: f1 = {m:.4f} ± {s:.4f}")
            log_dict.update({
                'iaf/signal_acc_mean':  iaf_acc_mean, 'iaf/signal_acc_std':  iaf_acc_std,
                'iaf/macro_f1_mean':    iaf_mf1_mean, 'iaf/macro_f1_std':    iaf_mf1_std,
                'iaf/patient_acc_mean': iaf_pat_mean, 'iaf/patient_acc_std': iaf_pat_std,
            })
            for cls in iaf_class_present:
                log_dict[f'iaf/{cls}_f1_mean'] = float(np.mean(fold_iaf_perclass[cls]))
                log_dict[f'iaf/{cls}_f1_std']  = float(np.std(fold_iaf_perclass[cls]))

        if fold_map_accs:
            map_acc_mean = float(np.mean(fold_map_accs))
            map_acc_std  = float(np.std(fold_map_accs))
            map_mf1_mean = float(np.mean(fold_map_macro_f1))
            map_mf1_std  = float(np.std(fold_map_macro_f1))
            print(f"  [CV] map_acc   : {map_acc_mean:.4f} ± {map_acc_std:.4f}  "
                  f"(folds: {[f'{v:.4f}' for v in fold_map_accs]})")
            print(f"  [CV] map_mf1   : {map_mf1_mean:.4f} ± {map_mf1_std:.4f}")
            log_dict.update({
                'clf/map_acc_mean':      map_acc_mean,
                'clf/map_acc_std':       map_acc_std,
                'clf/map_macro_f1_mean': map_mf1_mean,
                'clf/map_macro_f1_std':  map_mf1_std,
            })

        # ── LR aggregate ────────────────────────────────────────────────────────
        if use_logreg and fold_lr_accs:
            lr_acc_mean  = float(np.mean(fold_lr_accs));   lr_acc_std  = float(np.std(fold_lr_accs))
            lr_mf1_mean  = float(np.mean(fold_lr_macro_f1)); lr_mf1_std = float(np.std(fold_lr_macro_f1))
            print(f"\n  [CV-LR] accuracy : {lr_acc_mean:.4f} ± {lr_acc_std:.4f}")
            print(f"  [CV-LR] macro_f1 : {lr_mf1_mean:.4f} ± {lr_mf1_std:.4f}")
            for cls in class_names:
                m = float(np.mean(fold_lr_perclass[cls]))
                s = float(np.std(fold_lr_perclass[cls]))
                print(f"  [CV-LR] {cls:<12}: f1 = {m:.4f} ± {s:.4f}")
            log_dict.update({
                'clf/lr_test_acc_mean':  lr_acc_mean,  'clf/lr_test_acc_std':  lr_acc_std,
                'clf/lr_macro_f1_mean':  lr_mf1_mean,  'clf/lr_macro_f1_std':  lr_mf1_std,
            })
            for cls in class_names:
                log_dict[f'clf/lr_{cls}_f1-score_mean'] = float(np.mean(fold_lr_perclass[cls]))
                log_dict[f'clf/lr_{cls}_f1-score_std']  = float(np.std(fold_lr_perclass[cls]))

        # ── Aggregated confusion matrices (sum over all folds) ───────────────────
        sz = (max(6, len(class_names) + 2), max(5, len(class_names) + 1))
        agg_labels_arr = np.array(agg_labels)
        agg_preds_arr  = np.array(agg_preds)

        cm_sig = confusion_matrix(agg_labels_arr, agg_preds_arr)
        fig_cm_sig, ax_sig = plt.subplots(figsize=sz)
        ConfusionMatrixDisplay(confusion_matrix=cm_sig,
                               display_labels=class_names).plot(
            ax=ax_sig, colorbar=True, cmap='Blues', xticks_rotation=45)
        ax_sig.set_title(f"CV Signal-level Confusion Matrix – {pth_stem}\n"
                         f"acc={acc_mean:.4f} ± {acc_std:.4f}")
        plt.tight_layout()

        fig_cm_lr = None
        if use_logreg and agg_lr_preds:
            cm_lr = confusion_matrix(np.array(agg_labels), np.array(agg_lr_preds))
            fig_cm_lr, ax_lr = plt.subplots(figsize=sz)
            ConfusionMatrixDisplay(confusion_matrix=cm_lr,
                                   display_labels=class_names).plot(
                ax=ax_lr, colorbar=True, cmap='Greens', xticks_rotation=45)
            ax_lr.set_title(f"CV LogReg Confusion Matrix – {pth_stem}\n"
                            f"acc={lr_acc_mean:.4f} ± {lr_acc_std:.4f}")
            plt.tight_layout()

        fig_cm_map = None
        if agg_map_labels:
            cm_map = confusion_matrix(np.array(agg_map_labels), np.array(agg_map_preds))
            fig_cm_map, ax_map = plt.subplots(figsize=sz)
            ConfusionMatrixDisplay(confusion_matrix=cm_map,
                                   display_labels=class_names).plot(
                ax=ax_map, colorbar=True, cmap='Purples', xticks_rotation=45)
            ax_map.set_title(f"CV Map-level Confusion Matrix – {pth_stem}\n"
                             f"acc={map_acc_mean:.4f} ± {map_acc_std:.4f}"
                             if fold_map_accs else
                             f"CV Map-level Confusion Matrix – {pth_stem}")
            plt.tight_layout()

        # ── Hierarchical 2-stage classification ─────────────────────────────────
        if hierarchical:
            print(f"\n{'='*60}")
            print(f"[HIER] 2-Stage Hierarchical Classification")
            print(f"  Stage 1: Arrhythmia (AF+Flutter) vs SR (SR/SR300/SR600/Other_SR)")
            print(f"  Stage 2: AF vs Flutter (arrhythmia signals only)")

            y1_all   = np.array([1 if r in ARRHYTHMIA_LABELS else 0 for r in rhy_all])
            s1_names = np.array(['SR', 'Arrhythmia'])
            arr_all  = np.array([r in ARRHYTHMIA_LABELS for r in rhy_all])
            print(f"  Stage 1 distribution: Arrhythmia={arr_all.sum():,}  SR={(~arr_all).sum():,}")

            le2 = LabelEncoder(); le2.fit(rhy_all[arr_all])
            s2_names = le2.classes_
            y2_all   = np.full(len(rhy_all), -1, dtype=int)
            y2_all[arr_all] = le2.transform(rhy_all[arr_all])
            print(f"  Stage 2 classes: {list(s2_names)}")

            # IAF stage 2 ground truth
            y_iaf2_enc = None; X_iaf2 = None; pids_iaf2 = None
            if X_iaf is not None and rhy_iaf_orig is not None:
                iaf_arr_valid = np.array([r in le2.classes_ for r in rhy_iaf_orig])
                if iaf_arr_valid.sum() > 0:
                    y_iaf2_enc = le2.transform(rhy_iaf_orig[iaf_arr_valid])
                    X_iaf2     = X_iaf[iaf_arr_valid]
                    pids_iaf2  = pids_iaf[iaf_arr_valid]

            fold_s1_accs = []; fold_s1_mf1 = []
            fold_s1_perclass = {'SR': [], 'Arrhythmia': []}
            fold_s2_accs = []; fold_s2_mf1 = []
            fold_s2_perclass  = {c: [] for c in s2_names}
            fold_iaf_s1_accs  = []; fold_iaf_s2_accs = []; fold_iaf_s2_mf1 = []
            fold_iaf_s2_perclass = {c: [] for c in s2_names}

            for fold_idx, (test_pids, train_pids) in enumerate(folds):
                print(f"\n{'─'*60}")
                print(f"  [HIER Fold {fold_idx+1}/{n_folds}]")
                te_mask = np.isin(pids_all, list(test_pids))
                tr_mask = ~te_mask
                X_tr = X_all[tr_mask]; X_te = X_all[te_mask]
                pids_tr = pids_all[tr_mask]

                # ── Stage 1: Arrhythmia vs SR ────────────────────────────────────
                print(f"\n  Stage 1: Arrhythmia vs SR")
                clf1, preds1, labels1, acc1, mu1, std1, _ = train_classifier(
                    X_tr, y1_all[tr_mask], X_te, y1_all[te_mask],
                    patient_ids_train=pids_tr,
                    clf_arch=clf_arch, hidden_dim=mlp_hidden_dim, dropout=mlp_dropout,
                    lr=mlp_lr, weight_decay=mlp_weight_decay, epochs=mlp_epochs,
                    batch_size=batch_size, val_frac=mlp_val_frac, patience=mlp_patience,
                    es_criterion=mlp_es_criterion, seed=seed+fold_idx, device=str(device_obj))
                rep1  = classification_report(labels1, preds1, target_names=s1_names,
                                              output_dict=True, digits=4, zero_division=0)
                mf1_1 = float(rep1['macro avg']['f1-score'])
                fold_s1_accs.append(float(acc1)); fold_s1_mf1.append(mf1_1)
                for cls in ('SR', 'Arrhythmia'):
                    fold_s1_perclass[cls].append(float(rep1.get(cls, {}).get('f1-score', 0.0)))
                print(f"  Stage 1 fold {fold_idx+1}: acc={acc1:.4f}  macro_f1={mf1_1:.4f}")
                fig1, ax1 = plt.subplots(figsize=(5, 4))
                ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(labels1, preds1),
                                       display_labels=s1_names).plot(
                    ax=ax1, colorbar=True, cmap='Blues', xticks_rotation=45)
                ax1.set_title(f"HIER Stage 1 Fold {fold_idx+1} – {pth_stem}\nacc={acc1:.4f}")
                plt.tight_layout()
                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({f'hier/stage1_cm_fold{fold_idx+1}': wandb.Image(fig1)})
                plt.close(fig1)
                _save_clf_model(clf1, f'hier_stage1_fold{fold_idx+1}_{clf_arch}',
                                class_names=list(s1_names), X_train_shape1=X_tr.shape[1],
                                mlp_hidden_dim=mlp_hidden_dim, signal_type=signal_type,
                                pth_stem=pth_stem, mu=mu1, std=std1)

                # ── Stage 2: AF vs Flutter ───────────────────────────────────────
                arr_tr = arr_all[tr_mask]; arr_te = arr_all[te_mask]
                X2_tr  = X_tr[arr_tr];  y2_tr = y2_all[tr_mask][arr_tr]
                X2_te  = X_te[arr_te];  y2_te = y2_all[te_mask][arr_te]
                clf2 = mu2 = std2 = None
                if len(X2_tr) > 0 and len(np.unique(y2_tr)) > 1 and len(X2_te) > 0:
                    print(f"\n  Stage 2: AF vs Flutter ({arr_tr.sum()} tr / {arr_te.sum()} te arrhythmia signals)")
                    clf2, preds2, labels2, acc2, mu2, std2, _ = train_classifier(
                        X2_tr, y2_tr, X2_te, y2_te,
                        patient_ids_train=pids_tr[arr_tr],
                        clf_arch=clf_arch, hidden_dim=mlp_hidden_dim, dropout=mlp_dropout,
                        lr=mlp_lr, weight_decay=mlp_weight_decay, epochs=mlp_epochs,
                        batch_size=batch_size, val_frac=mlp_val_frac, patience=mlp_patience,
                        es_criterion=mlp_es_criterion, seed=seed+fold_idx+100, device=str(device_obj))
                    rep2  = classification_report(labels2, preds2, target_names=s2_names,
                                                  output_dict=True, digits=4, zero_division=0)
                    mf1_2 = float(rep2['macro avg']['f1-score'])
                    fold_s2_accs.append(float(acc2)); fold_s2_mf1.append(mf1_2)
                    for cls in s2_names:
                        fold_s2_perclass[cls].append(float(rep2.get(cls, {}).get('f1-score', 0.0)))
                    print(f"  Stage 2 fold {fold_idx+1}: acc={acc2:.4f}  macro_f1={mf1_2:.4f}")
                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(labels2, preds2),
                                           display_labels=s2_names).plot(
                        ax=ax2, colorbar=True, cmap='Oranges', xticks_rotation=45)
                    ax2.set_title(f"HIER Stage 2 Fold {fold_idx+1} – {pth_stem}\nacc={acc2:.4f}")
                    plt.tight_layout()
                    if WANDB_AVAILABLE and wandb.run is not None:
                        wandb.log({f'hier/stage2_cm_fold{fold_idx+1}': wandb.Image(fig2)})
                    plt.close(fig2)
                    _save_clf_model(clf2, f'hier_stage2_fold{fold_idx+1}_{clf_arch}',
                                    class_names=list(s2_names), X_train_shape1=X2_tr.shape[1],
                                    mlp_hidden_dim=mlp_hidden_dim, signal_type=signal_type,
                                    pth_stem=pth_stem, mu=mu2, std=std2)

                # ── IAF hierarchical evaluation ──────────────────────────────────
                if X_iaf is not None and y_iaf2_enc is not None and clf2 is not None:
                    X_iaf_n1   = (X_iaf - mu1) / std1
                    preds_iaf1 = _clf_predict(clf1, X_iaf_n1, str(device_obj), batch_size)
                    acc_iaf1   = float(accuracy_score(np.ones(len(X_iaf), dtype=int), preds_iaf1))
                    fold_iaf_s1_accs.append(acc_iaf1)
                    print(f"  IAF Stage 1 fold {fold_idx+1}: recall_arrhythmia={acc_iaf1:.4f}  "
                          f"({int((preds_iaf1==1).sum())}/{len(X_iaf)} predicted arrhythmia)")

                    X_iaf2_n   = (X_iaf2 - mu2) / std2
                    preds_iaf2 = _clf_predict(clf2, X_iaf2_n, str(device_obj), batch_size)
                    acc_iaf2   = float(accuracy_score(y_iaf2_enc, preds_iaf2))
                    rep_iaf2   = classification_report(y_iaf2_enc, preds_iaf2,
                                                       target_names=s2_names, output_dict=True,
                                                       digits=4, zero_division=0)
                    mf1_iaf2   = float(rep_iaf2['macro avg']['f1-score'])
                    fold_iaf_s2_accs.append(acc_iaf2); fold_iaf_s2_mf1.append(mf1_iaf2)
                    for cls in s2_names:
                        fold_iaf_s2_perclass[cls].append(
                            float(rep_iaf2.get(cls, {}).get('f1-score', 0.0)))
                    iaf2_pat_p, iaf2_pat_l, _ = _clf_per_map(
                        preds_iaf2, y_iaf2_enc, pids_iaf2, s2_names)
                    iaf2_pat_acc = float(accuracy_score(iaf2_pat_l, iaf2_pat_p))
                    print(f"  IAF Stage 2 fold {fold_idx+1}: signal_acc={acc_iaf2:.4f}  "
                          f"macro_f1={mf1_iaf2:.4f}  patient_acc={iaf2_pat_acc:.4f}")
                    fig_iaf2, ax_iaf2 = plt.subplots(figsize=(5, 4))
                    ConfusionMatrixDisplay(
                        confusion_matrix=confusion_matrix(y_iaf2_enc, preds_iaf2),
                        display_labels=s2_names).plot(
                        ax=ax_iaf2, colorbar=True, cmap='Purples', xticks_rotation=45)
                    ax_iaf2.set_title(f"IAF HIER S2 Fold {fold_idx+1}\n"
                                      f"acc={acc_iaf2:.4f}  pat_acc={iaf2_pat_acc:.4f}")
                    plt.tight_layout()
                    if WANDB_AVAILABLE and wandb.run is not None:
                        wandb.log({f'hier/iaf_stage2_cm_fold{fold_idx+1}': wandb.Image(fig_iaf2)})
                    plt.close(fig_iaf2)

            # ── Hierarchical aggregate ───────────────────────────────────────────
            print(f"\n{'='*60}")
            s1m = float(np.mean(fold_s1_accs)); s1s = float(np.std(fold_s1_accs))
            s1f = float(np.mean(fold_s1_mf1));  s1fs = float(np.std(fold_s1_mf1))
            print(f"  [HIER Stage 1] acc={s1m:.4f}±{s1s:.4f}  macro_f1={s1f:.4f}±{s1fs:.4f}")
            for cls in ('SR', 'Arrhythmia'):
                m = float(np.mean(fold_s1_perclass[cls])); s = float(np.std(fold_s1_perclass[cls]))
                print(f"  [HIER S1] {cls:<12}: f1={m:.4f}±{s:.4f}")
            log_dict.update({
                'hier/s1_acc_mean': s1m, 'hier/s1_acc_std': s1s,
                'hier/s1_mf1_mean': s1f, 'hier/s1_mf1_std': s1fs,
            })
            if fold_s2_accs:
                s2m = float(np.mean(fold_s2_accs)); s2s = float(np.std(fold_s2_accs))
                s2f = float(np.mean(fold_s2_mf1));  s2fs = float(np.std(fold_s2_mf1))
                print(f"  [HIER Stage 2] acc={s2m:.4f}±{s2s:.4f}  macro_f1={s2f:.4f}±{s2fs:.4f}")
                for cls in s2_names:
                    m = float(np.mean(fold_s2_perclass[cls])); s = float(np.std(fold_s2_perclass[cls]))
                    print(f"  [HIER S2] {cls:<12}: f1={m:.4f}±{s:.4f}")
                log_dict.update({
                    'hier/s2_acc_mean': s2m, 'hier/s2_acc_std': s2s,
                    'hier/s2_mf1_mean': s2f, 'hier/s2_mf1_std': s2fs,
                })
                for cls in s2_names:
                    log_dict[f'hier/s2_{cls}_f1_mean'] = float(np.mean(fold_s2_perclass[cls]))
                    log_dict[f'hier/s2_{cls}_f1_std']  = float(np.std(fold_s2_perclass[cls]))
            if fold_iaf_s1_accs:
                iaf1m = float(np.mean(fold_iaf_s1_accs)); iaf1s = float(np.std(fold_iaf_s1_accs))
                print(f"  [HIER IAF S1] recall_arrhythmia={iaf1m:.4f}±{iaf1s:.4f}")
                log_dict.update({'hier/iaf_s1_recall_mean': iaf1m, 'hier/iaf_s1_recall_std': iaf1s})
            if fold_iaf_s2_accs:
                iaf2m = float(np.mean(fold_iaf_s2_accs)); iaf2s = float(np.std(fold_iaf_s2_accs))
                iaf2f = float(np.mean(fold_iaf_s2_mf1));  iaf2fs = float(np.std(fold_iaf_s2_mf1))
                print(f"  [HIER IAF S2] acc={iaf2m:.4f}±{iaf2s:.4f}  macro_f1={iaf2f:.4f}±{iaf2fs:.4f}")
                for cls in s2_names:
                    m = float(np.mean(fold_iaf_s2_perclass[cls]))
                    s = float(np.std(fold_iaf_s2_perclass[cls]))
                    print(f"  [HIER IAF S2] {cls:<12}: f1={m:.4f}±{s:.4f}")
                log_dict.update({
                    'hier/iaf_s2_acc_mean': iaf2m, 'hier/iaf_s2_acc_std': iaf2s,
                    'hier/iaf_s2_mf1_mean': iaf2f, 'hier/iaf_s2_mf1_std': iaf2fs,
                })
                for cls in s2_names:
                    log_dict[f'hier/iaf_s2_{cls}_f1_mean'] = float(np.mean(fold_iaf_s2_perclass[cls]))
                    log_dict[f'hier/iaf_s2_{cls}_f1_std']  = float(np.std(fold_iaf_s2_perclass[cls]))

    # ── t-SNE ───────────────────────────────────────────────────────────────
    if do_tsne:
        print(f"\n[t-SNE] Computing 2-D projection (up to 10,000 per rhythm) ...")
        rng     = np.random.default_rng(seed)
        idx_t   = []
        for r in np.unique(rhy_all):
            r_idx = np.where(rhy_all == r)[0]
            n     = min(10000, len(r_idx))
            idx_t.append(rng.choice(r_idx, size=n, replace=False))
        idx_t  = np.concatenate(idx_t)
        X_sub  = X_all[idx_t]
        r_sub  = rhy_all[idx_t]
        p_sub  = pids_all[idx_t]
        print(f"  {len(idx_t):,} points total — " +
              ", ".join(f"{r}:{(r_sub==r).sum():,}" for r in sorted(np.unique(r_sub))))
        emb    = TSNE(n_components=2, perplexity=30, random_state=seed,
                      n_jobs=-1).fit_transform(X_sub)

        fig_tsne, ax = plt.subplots(figsize=(8, 7))
        for r in sorted(np.unique(r_sub)):
            m     = r_sub == r
            color = RHYTHM_COLORS.get(r, RHYTHM_COLORS['unknown'])
            ax.scatter(emb[m, 0], emb[m, 1], c=color, label=r, s=6, alpha=0.5)
        ax.set_title(f"t-SNE — latent space by rhythm\n{pth_stem}")
        ax.legend(markerscale=3)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        plt.tight_layout()

        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({'clf/tsne': wandb.Image(fig_tsne)})
        plt.close(fig_tsne)

    # ── Per-signal / per-patient recon/denoise on ALL test patients ──────────
    if not no_recon:
        print(f"\n--- Recon metrics (all {len(np.unique(pids_all))} patients, {len(sigs_all)} señales) ---")
        _recons_cv = _reconstruct(ae_model, sigs_all, batch_size, str(device_obj))
        clin_patient_rows_cv, clin_summary_cv = _clinical_metrics(
            sigs_all, _recons_cv, pids_all,
            is_bipolar=is_bipolar, fs=sampling_freq, p_inf=p_inf, p_sup=p_sup)

        print(f"\n--- Denoise metrics (all {len(np.unique(pids_all))} patients, {len(sigs_all)} señales) ---")
        _sigs_all_t  = torch.from_numpy(sigs_all).float()
        np.random.seed(42)   # fixed seed → same noise for every model
        torch.manual_seed(42)
        _noisy_all_t = add_combined_noise(
            _sigs_all_t,
            noise_types = ['gaussian', 'baseline_wander', 'powerline', 'spike'],
            snr_db_min  = noise_snr_min,
            snr_db_max  = noise_snr_max,
            fs          = sampling_freq,
        )
        _denoised_cv = _reconstruct(ae_model, _noisy_all_t.numpy(), batch_size, str(device_obj))
        clin_dn_patient_rows_cv, clin_dn_summary_cv = _clinical_metrics(
            sigs_all, _denoised_cv, pids_all,
            is_bipolar=is_bipolar, fs=sampling_freq, p_inf=p_inf, p_sup=p_sup)

    # ── Reconstruction visualizations ─────────────────────────────────────────
    if viz_args is not None and WANDB_AVAILABLE and wandb.run is not None:
        _data_t = torch.from_numpy(sigs_all).float()
        _p_inf_viz = p_inf if p_inf is not None else 0.0
        _p_sup_viz = p_sup if p_sup is not None else 1.0
        visualize_test_with_noise_info(
            ae_model, _data_t, _p_inf_viz, _p_sup_viz,
            label='test', num_samples=15,
            args=viz_args, device=str(device_obj),
            is_bipolar=is_bipolar,
        )

    if WANDB_AVAILABLE and wandb.run is not None:
        if not no_clf:
            wandb.log(log_dict)
            for k, v in log_dict.items():
                wandb.run.summary[k] = v
            _log_wandb_table(all_patient_rows,  'clf/per_patient')
            _log_wandb_table(fold_assign_rows,  'clf/cv_fold_assignments')
            if fig_cm_sig is not None:
                wandb.log({'clf/cv_confusion_matrix_signal': wandb.Image(fig_cm_sig)})
            if fig_cm_lr is not None:
                wandb.log({'clf/cv_confusion_matrix_lr': wandb.Image(fig_cm_lr)})
            if fig_cm_map is not None:
                wandb.log({'clf/cv_confusion_matrix_map': wandb.Image(fig_cm_map)})
        if not no_recon:
            _log_wandb_table(clin_patient_rows_cv,    'recon/per_patient')
            wandb.log({f'recon/{k}':   v for k, v in clin_summary_cv.items()})
            _log_wandb_table(clin_dn_patient_rows_cv, 'denoise/per_patient')
            wandb.log({f'denoise/{k}': v for k, v in clin_dn_summary_cv.items()})

    if not no_clf:
        if fig_cm_sig is not None:
            plt.close(fig_cm_sig)
        if fig_cm_lr is not None:
            plt.close(fig_cm_lr)
        if fig_cm_map is not None:
            plt.close(fig_cm_map)

    return log_dict if not no_clf else {}


def _run_one(pth_path: str, args) -> dict:
    """Run evaluate_clf for a single .pth, managing the WandB run lifecycle."""
    pth_stem = os.path.splitext(os.path.basename(pth_path))[0]
    no_clf   = getattr(args, 'no_clf', False)
    if getattr(args, 'viz_only', False):
        args.reconstruction = True  # viz_only implies reconstruction (for viz_args)

    if not args.no_wandb and WANDB_AVAILABLE:
        suffix  = '_recon' if no_clf else '_cv'
        wandb.init(
            project = args.wandb_project,
            entity  = args.wandb_entity,
            name    = pth_stem + suffix,
            config  = vars(args),
            reinit  = True,
        )

    cv_result = run_cv_evaluate(
        pth_path         = pth_path,
        data_dir         = args.data_dir,
        dataset_path     = getattr(args, 'dataset_path', None),
        n_folds          = getattr(args, 'cv_folds', 5),
        seed             = args.seed,
        latent_dim       = args.latent_dim,
        filters_initial  = args.filters_initial,
        dense_dim        = args.dense_dim,
        q_parameter      = args.q_parameter,
        dropout_rate     = args.dropout_rate,
        loss_function    = args.loss_function,
        clf_arch         = args.clf_arch,
        mlp_hidden_dim   = args.mlp_hidden_dim,
        mlp_lr           = args.mlp_lr,
        mlp_epochs       = args.mlp_epochs,
        mlp_dropout      = args.mlp_dropout,
        mlp_weight_decay = args.mlp_weight_decay,
        mlp_val_frac         = args.mlp_val_frac,
        mlp_patience         = args.mlp_patience,
        mlp_es_criterion     = args.mlp_es_criterion,
        map_signals_per_pred = args.map_signals_per_pred,
        use_logreg           = getattr(args, 'logreg', False),
        batch_size           = args.batch_size,
        device               = str(torch.device(args.device
                                                if torch.cuda.is_available()
                                                else 'cpu')),
        noise_snr_min        = args.noise_snr_min,
        noise_snr_max        = args.noise_snr_max,
        sampling_freq        = args.sampling_freq,
        no_clf               = no_clf,
        no_recon             = getattr(args, 'no_recon', False),
        viz_args             = args if getattr(args, 'reconstruction', False) else None,
        viz_only             = getattr(args, 'viz_only', False),
        do_tsne              = getattr(args, 'tsne', False),
        iaf_h5               = getattr(args, 'iaf_h5', None),
        hierarchical         = getattr(args, 'hierarchical', False),
        raw_signals          = getattr(args, 'raw_signals', False),
    )

    if not args.no_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()

    return {
        'pth_stem':   pth_stem,
        'signal_acc': cv_result.get('clf/test_acc_mean'),
        'map_acc':    cv_result.get('clf/map_acc_mean'),
    }


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            'Unified CLF evaluation: MLP/DeepMLP on latent space '
            '(clf_train/clf_test), with reconstruction metrics, t-SNE, '
            'map-level evaluation, misclassified viewer, and model saving.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── pth path(s) ──────────────────────────────────────────────────────────
    pth_grp = p.add_mutually_exclusive_group(required=True)
    pth_grp.add_argument('--pth_path', nargs='+', default=None,
                         help='Path(s) to autoencoder .pth checkpoint(s). '
                              'Accepts glob patterns via shell expansion.')
    pth_grp.add_argument('--pth_dir', default=None,
                         help='Directory with .pth files — runs evaluation for '
                              'every .pth found (sorted). Mutually exclusive '
                              'with --pth_path.')

    # ── data ─────────────────────────────────────────────────────────────────
    p.add_argument('--data_dir',     type=str, default=None,
                   help='Preprocessed data root dir. Signal type inferred from pth stem.')
    p.add_argument('--dataset_path', type=str, default=None,
                   help='Explicit HDF5 dataset directory. Overrides --data_dir.')

    # ── clf split settings ────────────────────────────────────────────────────
    p.add_argument('--clf_train_name', type=str, default='clf_train',
                   help='Stem of the train split h5 file (without .h5). '
                        'Test file derived by replacing "train" → "test".')
    p.add_argument('--extra_test_split', type=str, default=None,
                   help='Additional split to evaluate after training (e.g. "train"). '
                        'Loads {split}.h5 and reports accuracy + confusion matrix.')

    # ── model architecture overrides ──────────────────────────────────────────
    p.add_argument('--loss_function',    type=str, default=None,
                   choices=['mse', 'dtw'],
                   help='Override loss function. If None, inferred automatically from pth filename.')
    p.add_argument('--latent_dim',       type=int,   default=64)
    p.add_argument('--filters_initial',  type=int,   default=64)
    p.add_argument('--dense_dim',        type=int,   default=128)
    p.add_argument('--q_parameter',      type=int,   default=2)
    p.add_argument('--dropout_rate',     type=float, default=0.1)

    # ── classifier ────────────────────────────────────────────────────────────
    p.add_argument('--clf_pth', type=str, default=None,
                   help='Path to a pre-trained classifier .pth (from clf_model_pth/). '
                        'If provided, skips training and runs inference only.')
    p.add_argument('--n_seeds', type=int, default=1,
                   help='Train this many classifiers with consecutive seeds '
                        '(seed, seed+1, …) and average their softmax outputs. '
                        'Each model is saved separately. Default: 1 (no ensemble).')
    p.add_argument('--clf_save_dir', type=str, default=None,
                   help='Directory to save trained classifier .pth file(s). '
                        'Default: clf_model_pth/{signal_type}/.')
    p.add_argument('--map_signals_per_pred', type=int, default=0,
                   help='Chunk map signals into blocks of this size for majority vote. '
                        'Remainder signals (< block size) are discarded. '
                        '0 = use all signals at once (default).')
    p.add_argument('--clf_arch', type=str, default='mlp',
                   choices=['mlp', 'deep_mlp'],
                   help='Classifier architecture.')
    p.add_argument('--mlp_hidden_dim',   type=int,   default=128)
    p.add_argument('--mlp_lr',           type=float, default=1e-3)
    p.add_argument('--mlp_epochs',       type=int,   default=200)
    p.add_argument('--mlp_dropout',      type=float, default=0.2)
    p.add_argument('--mlp_weight_decay', type=float, default=1e-3,
                   help='L2 regularisation for Adam.')
    p.add_argument('--mlp_val_frac',     type=float, default=0.2,
                   help='Fraction of train patients held out as val for early stopping.')
    p.add_argument('--mlp_patience',     type=int,   default=20,
                   help='Early stopping patience (epochs).')
    p.add_argument('--mlp_es_criterion', type=str,   default='val_loss',
                   choices=['val_acc', 'val_loss'],
                   help='Early stopping criterion.')

    # ── noise / denoising ─────────────────────────────────────────────────────
    p.add_argument('--noise_snr_min',  type=float, default=-5.0,
                   help='Min SNR (dB) for denoising power evaluation.')
    p.add_argument('--noise_snr_max',  type=float, default=10.0,
                   help='Max SNR (dB) for denoising power evaluation.')
    p.add_argument('--sampling_freq',  type=int,   default=500)

    # ── feature flags ─────────────────────────────────────────────────────────
    p.add_argument('--reconstruction', action='store_true',
                   help='Run reconstruction metrics (R², MSE, Vpp, DF, NLEO, denoising).')
    p.add_argument('--viz_only', action='store_true',
                   help='Skip all metrics; only generate reconstruction+noise plots (implies --reconstruction).')
    p.add_argument('--tsne',           action='store_true',
                   help='Run t-SNE visualisation of test latent vectors.')
    p.add_argument('--hierarchical', action='store_true',
                   help='After the standard CV, run a 2-stage hierarchical classifier: '
                        'Stage 1 = Arrhythmia (AF+Flutter) vs SR; '
                        'Stage 2 = AF vs Flutter on predicted arrhythmias. '
                        'Saves Stage 1 and Stage 2 .pth per fold to clf_model_pth/.')
    p.add_argument('--raw_signals', action='store_true',
                   help='Classify using flattened raw signals instead of AE latent vectors. '
                        'Baseline: no AE encoding needed. Combine with --no_recon to skip AE loading.')
    p.add_argument('--iaf_h5', type=str, default=None,
                   help='Path to pre-built IAF h5 (build_iaf_dataset.py). '
                        'When provided, each CV fold classifier is also evaluated '
                        'on the IAF signals (AF and AFL→Flutter). '
                        'Metrics logged as iaf/signal_acc, iaf/patient_acc, etc.')
    p.add_argument('--show_misclassified', action='store_true',
                   help='Generate and log the Plotly misclassified signals viewer to WandB.')
    p.add_argument('--max_signals',  type=int, default=500,
                   help='Max misclassified signals embedded in the HTML viewer.')

    # ── misc ──────────────────────────────────────────────────────────────────
    p.add_argument('--batch_size', type=int,   default=256)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--device',     type=str,   default='cuda',
                   choices=['cuda', 'cpu'])
    p.add_argument('--output_dir', type=str,   default='results_clf_pipeline',
                   help='Output directory (reserved for future use).')

    # ── WandB ─────────────────────────────────────────────────────────────────
    p.add_argument('--wandb_project', type=str, default='autoencoder-egms-fail')
    p.add_argument('--wandb_entity',  type=str, default=None)
    p.add_argument('--no_wandb',      action='store_true')

    # ── Cross-validation ──────────────────────────────────────────────────────
    p.add_argument('--no_cv',    action='store_true',
                   help='Use clf_train/clf_test splits instead of k-fold CV (CV is default).')
    p.add_argument('--cv_folds', type=int, default=5,
                   help='Number of folds for CV mode (default: 5).')
    p.add_argument('--logreg',   action='store_true',
                   help='Also train Logistic Regression baseline per fold in CV mode.')
    p.add_argument('--no_clf',   action='store_true',
                   help='Skip classification entirely (only reconstruction if --reconstruction set).')
    p.add_argument('--no_recon', action='store_true',
                   help='Skip reconstruction and denoising metrics (only classification).')

    return p.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.data_dir is None and args.dataset_path is None:
        print("ERROR: provide --data_dir or --dataset_path.")
        sys.exit(1)

    # ── Resolve list of pth files ─────────────────────────────────────────────
    if args.pth_dir is not None:
        if not os.path.isdir(args.pth_dir):
            print(f"ERROR: --pth_dir not found: {args.pth_dir}")
            sys.exit(1)
        pth_list = sorted(_glob_module.glob(os.path.join(args.pth_dir, '*.pth')))
        if not pth_list:
            print(f"ERROR: No .pth files found in {args.pth_dir}")
            sys.exit(1)
        print(f"\nFound {len(pth_list)} .pth files in {args.pth_dir}:")
        for pp in pth_list:
            print(f"  {os.path.basename(pp)}")
    else:
        pth_list = args.pth_path

    # ── Run each model ────────────────────────────────────────────────────────
    results = []
    for i, pth_path in enumerate(pth_list):
        if not os.path.isfile(pth_path):
            print(f"\nWARNING: .pth not found, skipping: {pth_path}")
            continue

        pth_stem = os.path.splitext(os.path.basename(pth_path))[0]
        print(f"\n{'='*70}")
        print(f"  [{i+1}/{len(pth_list)}]  {pth_stem}")
        print(f"{'='*70}")

        try:
            row = _run_one(pth_path, args)
            results.append(row)
        except Exception as e:
            import traceback
            print(f"  ERROR processing {pth_path}:")
            traceback.print_exc()
            results.append({'pth_stem': pth_stem,
                            'signal_acc': None, 'map_acc': None})

    # ── Summary table ─────────────────────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"  SUMMARY ({len(results)} models)")
        print(f"{'='*70}")
        print(f"  {'Model':<55}  {'sig_acc':>8}  {'map_acc':>8}")
        print(f"  {'-'*55}  {'-'*8}  {'-'*8}")
        for row in sorted(results,
                          key=lambda r: r['signal_acc'] or -1.0,
                          reverse=True):
            sa = f"{row['signal_acc']:.4f}" if row['signal_acc'] is not None else "ERROR"
            ma = (f"{row['map_acc']:.4f}" if row['map_acc'] is not None else "N/A")
            print(f"  {row['pth_stem']:<55}  {sa:>8}  {ma:>8}")

    print("\nDone.")


if __name__ == '__main__':
    main()
