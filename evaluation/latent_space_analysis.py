"""
Latent Space Analysis Script
==============================
Two main functions:
  1. visualize_latent_space: Extract latent representations from the encoder,
     run t-SNE, and generate coloured scatter plots (logged to WandB).
  2. classify_latent_space: Train a simple MLP on the latent space and evaluate
     with a confusion matrix. Also generates an interactive HTML viewer for
     misclassified signals (logged to WandB).

Shared utilities (model loading, dataset loading, constants) live in eval_utils.py.
Misclassified viewer logic lives in miss_classifier.py.

Usage example
-------------
python latent_space_analysis.py \\
    --pth_path model_pth/UNIPOLAR_CLARAE_SCM_GLU_dtw_noise.pth \\
    --data_dir processed_data_final \\
    --output_dir results_plots
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
from eval_utils import (
    RHYTHM_COLORS, RHYTHM_PRIORITY, EXCLUDED_RHYTHMS,
    MODEL_REGISTRY, SC_MODELS, ENCODE_MODELS,
    MLPClassifier,
    _strip_loss_noise_suffix, _arch_from_stem,
    _signal_type_from_stem, _find_dataset_path,
    build_model, _infer_hyperparams_from_state, load_model,
    _decode_str_array, _load_rhythm_map_from_json,
    load_test_dataset, extract_latent,
    PLOTLY_AVAILABLE, WANDB_AVAILABLE,
)

from plots.plot_misclassified import visualize_misclassified

try:
    import plotly.graph_objects as go
except ImportError:
    pass

try:
    import wandb
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Plotting helpers — matplotlib (static PNG)
# ---------------------------------------------------------------------------

def _scatter_plot(embedding: np.ndarray, rhythms: np.ndarray,
                  title: str, xlabel: str, ylabel: str,
                  elapsed: float):
    """Generic 2-D scatter coloured by rhythm. Returns fig (caller closes it)."""
    unique_rhythms = np.unique(rhythms)
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(12, 9))

    for rhythm in unique_rhythms:
        mask = rhythms == rhythm
        color = RHYTHM_COLORS.get(rhythm, '#888888')
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, alpha=0.6, s=12, label=rhythm, rasterized=True
        )

    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(f"{title}\n(time: {elapsed:.1f} s)", fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.3)

    handles = [
        mpatches.Patch(color=RHYTHM_COLORS.get(r, '#888888'), label=r)
        for r in unique_rhythms
    ]
    ax.legend(handles=handles, markerscale=2, frameon=True,
              fancybox=True, shadow=True, fontsize=12,
              loc='best').get_frame().set_alpha(0.85)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plotting helpers — Plotly (interactive HTML)
# ---------------------------------------------------------------------------

def _plotly_scatter_2d(embedding: np.ndarray, rhythms: np.ndarray,
                       title: str, elapsed: float) -> str:
    """Interactive 2-D scatter coloured by rhythm. Returns HTML string."""
    if not PLOTLY_AVAILABLE:
        return ''
    fig = go.Figure()
    for rhythm in np.unique(rhythms):
        mask = rhythms == rhythm
        color = RHYTHM_COLORS.get(rhythm, '#888888')
        fig.add_trace(go.Scatter(
            x=embedding[mask, 0].tolist(),
            y=embedding[mask, 1].tolist(),
            mode='markers',
            name=rhythm,
            marker=dict(color=color, size=4, opacity=0.65),
            hovertemplate=(
                f'<b>{rhythm}</b><br>'
                'x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
            ),
        ))
    fig.update_layout(
        title=dict(text=f"{title}  <sup>(time: {elapsed:.1f} s)</sup>", x=0.5),
        xaxis_title='Component 1',
        yaxis_title='Component 2',
        legend_title='Rhythm',
        template='plotly_white',
        width=950, height=700,
        font=dict(size=13),
    )
    return fig.to_html(include_plotlyjs='cdn')


def _plotly_scatter_3d(embedding: np.ndarray, rhythms: np.ndarray,
                       title: str, elapsed: float) -> str:
    """Interactive 3-D scatter coloured by rhythm. Returns HTML string."""
    if not PLOTLY_AVAILABLE:
        return ''
    fig = go.Figure()
    for rhythm in np.unique(rhythms):
        mask = rhythms == rhythm
        color = RHYTHM_COLORS.get(rhythm, '#888888')
        fig.add_trace(go.Scatter3d(
            x=embedding[mask, 0].tolist(),
            y=embedding[mask, 1].tolist(),
            z=embedding[mask, 2].tolist(),
            mode='markers',
            name=rhythm,
            marker=dict(color=color, size=3, opacity=0.6),
            hovertemplate=(
                f'<b>{rhythm}</b><br>'
                'x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>'
            ),
        ))
    fig.update_layout(
        title=dict(text=f"{title}  <sup>(time: {elapsed:.1f} s)</sup>", x=0.5),
        scene=dict(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3',
        ),
        legend_title='Rhythm',
        template='plotly_white',
        width=950, height=750,
        font=dict(size=13),
    )
    return fig.to_html(include_plotlyjs='cdn')


# ---------------------------------------------------------------------------
# Function 1: Visualise latent space (t-SNE)
# ---------------------------------------------------------------------------

def visualize_latent_space(args):
    """
    Load test data, extract latent representations, run t-SNE,
    and log scatter plots to WandB.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    print(f"\nLoading model '{args.model_class}' from: {args.pth_path}")
    model = load_model(
        pth_path        = args.pth_path,
        model_class     = args.model_class,
        latent_dim      = args.latent_dim,
        filters_initial = args.filters_initial,
        dropout_rate    = args.dropout_rate,
        dense_dim       = args.dense_dim,
        input_length    = args.input_length,
        device          = device,
        q_parameter     = args.q_parameter,
    )
    print("  Model loaded successfully.")

    signals, rhythms, _p_inf, _p_sup = load_test_dataset(
        dataset_path        = args.dataset_path,
        n_signals_per_class = args.n_signals_per_class,
        json_dir            = args.json_dir,
        split               = 'test',
    )

    print(f"\nExtracting latent vectors (batch_size={args.batch_size}) ...")
    t0 = time.time()
    latent = extract_latent(model, signals, args.model_class,
                            args.batch_size, device)
    print(f"  Latent shape: {latent.shape}  ({time.time()-t0:.1f} s)")

    pth_stem  = os.path.splitext(os.path.basename(args.pth_path))[0]
    use_wandb = not getattr(args, 'no_wandb', True) and WANDB_AVAILABLE and wandb.run is not None

    # ------ t-SNE 2-D ------
    print(f"\nRunning t-SNE 2D  (perplexity={args.tsne_perplexity}, "
          f"n_iter={args.tsne_n_iter}) ...")
    t0 = time.time()
    tsne = TSNE(
        n_components = 2,
        perplexity   = args.tsne_perplexity,
        max_iter     = args.tsne_n_iter,
        init         = 'pca',
        random_state = 42,
    )
    tsne_emb     = tsne.fit_transform(latent)
    elapsed_tsne = time.time() - t0
    print(f"  t-SNE 2D done in {elapsed_tsne:.1f} s")

    fig_tsne = _scatter_plot(
        embedding = tsne_emb,
        rhythms   = rhythms,
        title     = f"t-SNE – Latent Space ({args.model_class})",
        xlabel    = "t-SNE Component 1",
        ylabel    = "t-SNE Component 2",
        elapsed   = elapsed_tsne,
    )
    wandb_log_latent = {}
    if use_wandb:
        wandb_log_latent['latent/tsne_2d'] = wandb.Image(fig_tsne)
    plt.close(fig_tsne)
    plt.rcParams.update(plt.rcParamsDefault)

    if PLOTLY_AVAILABLE and not getattr(args, 'skip_interactive_tsne', False):
        html_2d = _plotly_scatter_2d(
            embedding = tsne_emb,
            rhythms   = rhythms,
            title     = f"t-SNE 2D – Latent Space ({args.model_class})",
            elapsed   = elapsed_tsne,
        )
        if use_wandb and html_2d:
            wandb_log_latent['latent/tsne_2d_interactive'] = wandb.Html(html_2d)

    # ------ t-SNE 3-D (optional) ------
    if PLOTLY_AVAILABLE and getattr(args, 'tsne_3d', False):
        print(f"\nRunning t-SNE 3D  (perplexity={args.tsne_perplexity}, "
              f"n_iter={args.tsne_n_iter}) ...")
        t0 = time.time()
        tsne3 = TSNE(
            n_components = 3,
            perplexity   = args.tsne_perplexity,
            max_iter     = args.tsne_n_iter,
            init         = 'pca',
            random_state = 42,
        )
        tsne3_emb     = tsne3.fit_transform(latent)
        elapsed_tsne3 = time.time() - t0
        print(f"  t-SNE 3D done in {elapsed_tsne3:.1f} s")
        html_3d = _plotly_scatter_3d(
            embedding = tsne3_emb,
            rhythms   = rhythms,
            title     = f"t-SNE 3D – Latent Space ({args.model_class})",
            elapsed   = elapsed_tsne3,
        )
        if use_wandb and html_3d:
            wandb_log_latent['latent/tsne_3d'] = wandb.Html(html_3d)
    elif getattr(args, 'tsne_3d', False) and not PLOTLY_AVAILABLE:
        print("\n  --tsne_3d requested but plotly not installed; skipping 3D.")

    if use_wandb and wandb_log_latent:
        wandb.log(wandb_log_latent)

    print(f"\nVisualization complete. Plots logged to WandB.")
    return latent, rhythms


# ---------------------------------------------------------------------------
# MLP training helpers
# ---------------------------------------------------------------------------

def train_mlp(model, loader, optimizer, criterion, device):
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
def eval_mlp(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        preds       = logits.argmax(1)
        correct    += (preds == y).sum().item()
        total      += len(y)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    return (total_loss / total, correct / total,
            np.concatenate(all_preds), np.concatenate(all_labels))


# ---------------------------------------------------------------------------
# Function 2: Classify latent space with MLP
# ---------------------------------------------------------------------------

def classify_latent_space(args):
    """
    Train a 1-hidden-layer MLP on the latent space, evaluate it on a held-out
    test split, and log results to WandB.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nLoading model for classification ...")
    model = load_model(
        pth_path        = args.pth_path,
        model_class     = args.model_class,
        latent_dim      = args.latent_dim,
        filters_initial = args.filters_initial,
        dropout_rate    = args.dropout_rate,
        dense_dim       = args.dense_dim,
        input_length    = args.input_length,
        device          = device,
        q_parameter     = args.q_parameter,
    )
    signals, rhythms, p_inf, p_sup = load_test_dataset(
        dataset_path        = args.dataset_path,
        n_signals_per_class = None,
        json_dir            = args.json_dir,
        split               = 'test',
    )
    latent = extract_latent(model, signals, args.model_class,
                            args.batch_size, device)

    le = LabelEncoder()
    y  = le.fit_transform(rhythms)
    class_names = le.classes_

    print(f"\nClasses: {list(class_names)}")

    all_idx = np.arange(len(latent))
    idx_trainval, idx_test = train_test_split(
        all_idx,
        test_size    = args.mlp_test_split,
        random_state = 42,
        stratify     = y,
    )
    X_trainval, X_test = latent[idx_trainval], latent[idx_test]
    y_trainval, y_test = y[idx_trainval],      y[idx_test]
    signals_test       = signals[idx_test]

    idx_train, idx_val = train_test_split(
        np.arange(len(X_trainval)),
        test_size    = args.mlp_val_split / (1 - args.mlp_test_split),
        random_state = 42,
        stratify     = y_trainval,
    )
    X_train, X_val = X_trainval[idx_train], X_trainval[idx_val]
    y_train, y_val = y_trainval[idx_train], y_trainval[idx_val]

    print(f"\n  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # ------ Logistic Regression baseline ------
    print(f"\nTraining Logistic Regression baseline ...")
    lr_clf = LogisticRegression(max_iter=1000, random_state=42)
    lr_clf.fit(X_trainval, y_trainval)
    lr_preds = lr_clf.predict(X_test)
    lr_acc   = (lr_preds == y_test).mean()
    print(f"  LR test accuracy: {lr_acc:.4f}")

    # ------ MLP training ------
    clf = MLPClassifier(
        input_dim  = X_train.shape[1],
        hidden_dim = args.mlp_hidden_dim,
        n_classes  = len(class_names),
        dropout    = args.mlp_dropout,
    ).to(device)

    optimizer = torch.optim.Adam(clf.parameters(), lr=args.mlp_lr)
    criterion = nn.CrossEntropyLoss()

    t_train = torch.tensor(X_train, dtype=torch.float32)
    t_val   = torch.tensor(X_val,   dtype=torch.float32)
    t_test  = torch.tensor(X_test,  dtype=torch.float32)
    lbl_train = torch.tensor(y_train, dtype=torch.long)
    lbl_val   = torch.tensor(y_val,   dtype=torch.long)
    lbl_test  = torch.tensor(y_test,  dtype=torch.long)

    train_loader = DataLoader(TensorDataset(t_train, lbl_train),
                              batch_size=256, shuffle=True)
    val_loader   = DataLoader(TensorDataset(t_val,   lbl_val),   batch_size=256)
    test_loader  = DataLoader(TensorDataset(t_test,  lbl_test),  batch_size=256)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc, best_state = 0.0, None

    print(f"\nTraining MLP ({args.mlp_epochs} epochs) ...")
    for epoch in range(1, args.mlp_epochs + 1):
        tr_loss, tr_acc = train_mlp(clf, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, _, _ = eval_mlp(clf, val_loader, criterion, device)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.clone() for k, v in clf.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.mlp_epochs:
            print(f"  Epoch {epoch:3d}/{args.mlp_epochs}  "
                  f"train_loss={tr_loss:.4f} val_loss={vl_loss:.4f}  "
                  f"val_acc={vl_acc:.4f}")

    clf.load_state_dict(best_state)
    _, test_acc, preds, labels = eval_mlp(clf, test_loader, criterion, device)
    print(f"\n  MLP test accuracy : {test_acc:.4f}")
    print(f"  LR  test accuracy : {lr_acc:.4f}")

    # ------ Save classifier .pth ------
    pth_stem    = os.path.splitext(os.path.basename(args.pth_path))[0]
    clf_pth_dir = os.path.join(args.output_dir, 'classifier_pth')
    os.makedirs(clf_pth_dir, exist_ok=True)
    clf_save_path = os.path.join(clf_pth_dir, f'{pth_stem}_classifier.pth')
    torch.save({
        'model_state_dict': clf.state_dict(),
        'class_names':      list(class_names),
        'input_dim':        int(latent.shape[1]),
        'hidden_dim':       args.mlp_hidden_dim,
        'dropout':          args.mlp_dropout,
        'n_classes':        len(class_names),
        'best_val_acc':     float(best_val_acc),
        'ae_model_class':   args.model_class,
        'ae_pth_path':      args.pth_path,
    }, clf_save_path)
    print(f"  Classifier saved -> {clf_save_path}")

    use_wandb = not getattr(args, 'no_wandb', True) and WANDB_AVAILABLE and wandb.run is not None

    wandb_log_clf = {}

    # ------ MLP confusion matrix → WandB only ------
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) + 2),
                                    max(5, len(class_names) + 1)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap='Blues', xticks_rotation=45)
    ax.set_title(
        f"Confusion Matrix – MLP Classifier\n"
        f"({args.model_class}, latent_dim={args.latent_dim}, "
        f"test_acc={test_acc:.4f})",
        fontweight='bold', pad=12
    )
    plt.tight_layout()
    if use_wandb:
        wandb_log_clf['clf/mlp_confusion_matrix'] = wandb.Image(fig)
    plt.close(fig)

    # ------ LR confusion matrix → WandB only ------
    cm_lr = confusion_matrix(y_test, lr_preds)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) + 2),
                                    max(5, len(class_names) + 1)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap='Greens', xticks_rotation=45)
    ax.set_title(
        f"Confusion Matrix – Logistic Regression (baseline)\n"
        f"({args.model_class}, latent_dim={args.latent_dim}, "
        f"test_acc={lr_acc:.4f})",
        fontweight='bold', pad=12
    )
    plt.tight_layout()
    if use_wandb:
        wandb_log_clf['clf/lr_confusion_matrix'] = wandb.Image(fig)
    plt.close(fig)

    # ------ Training curves → WandB only ------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, args.mlp_epochs + 1)
    axes[0].plot(epochs_range, history['train_loss'], label='train')
    axes[0].plot(epochs_range, history['val_loss'],   label='val')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Cross-Entropy Loss'); axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.4)
    axes[1].plot(epochs_range, history['train_acc'], label='train')
    axes[1].plot(epochs_range, history['val_acc'],   label='val')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy'); axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.4)
    plt.suptitle(f'MLP Training History – {args.model_class}', fontweight='bold')
    plt.tight_layout()
    if use_wandb:
        wandb_log_clf['clf/training_curves'] = wandb.Image(fig)
    plt.close(fig)

    # ------ Misclassified viewer → WandB only ------
    if not getattr(args, 'skip_misclassified', False):
        visualize_misclassified(
            signals_test = signals_test,
            labels       = labels,
            preds        = preds,
            class_names  = class_names,
            pth_stem     = pth_stem,
            max_per_class= 5,
            p_inf        = p_inf,
            p_sup        = p_sup,
        )

    # ------ Console report ------
    print("\n--- MLP Classification Report ---")
    print(classification_report(labels, preds, target_names=class_names, digits=4))
    print("--- LR Classification Report ---")
    print(classification_report(y_test, lr_preds, target_names=class_names, digits=4))

    # ------ WandB: single log call with plots + metrics ------
    if use_wandb:
        report_mlp = classification_report(labels, preds,
                                           target_names=class_names,
                                           output_dict=True, digits=4)
        report_lr  = classification_report(y_test, lr_preds,
                                           target_names=class_names,
                                           output_dict=True, digits=4)
        wandb_log_clf.update({
            'clf/mlp_test_acc':     float(test_acc),
            'clf/mlp_best_val_acc': float(best_val_acc),
            'clf/lr_test_acc':      float(lr_acc),
        })
        for cls in class_names:
            for metric in ('precision', 'recall', 'f1-score'):
                wandb_log_clf[f'clf/mlp_{cls}_{metric}'] = float(report_mlp[cls][metric])
                wandb_log_clf[f'clf/lr_{cls}_{metric}']  = float(report_lr[cls][metric])
        wandb_log_clf['clf/mlp_macro_f1'] = float(report_mlp['macro avg']['f1-score'])
        wandb_log_clf['clf/lr_macro_f1']  = float(report_lr['macro avg']['f1-score'])
        wandb.log(wandb_log_clf)
        for k, v in wandb_log_clf.items():
            if isinstance(v, float):
                wandb.run.summary[k] = v

    print(f"\nClassification complete. Classifier .pth saved to: {clf_pth_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Latent space visualisation and classification for EGM autoencoders',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--pth_path', type=str, nargs='+', required=True)
    parser.add_argument('--model_class', type=str, default=None,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--json_dir', type=str, default=None)
    parser.add_argument('--latent_dim',      type=int,   default=64)
    parser.add_argument('--filters_initial', type=int,   default=64)
    parser.add_argument('--dropout_rate',    type=float, default=0.1)
    parser.add_argument('--dense_dim',       type=int,   default=128)
    parser.add_argument('--input_length',    type=int,   default=1250)
    parser.add_argument('--n_signals_per_class', type=int, default=10000)
    parser.add_argument('--batch_size',      type=int,   default=256)
    parser.add_argument('--tsne_perplexity', type=float, default=200)
    parser.add_argument('--tsne_n_iter',     type=int,   default=1000)
    parser.add_argument('--tsne_3d',         action='store_true')
    parser.add_argument('--q_parameter',     type=int,   default=2)
    parser.add_argument('--mlp_hidden_dim',  type=int,   default=256)
    parser.add_argument('--mlp_lr',          type=float, default=1e-3)
    parser.add_argument('--mlp_epochs',      type=int,   default=100)
    parser.add_argument('--mlp_dropout',     type=float, default=0.3)
    parser.add_argument('--mlp_val_split',   type=float, default=0.15)
    parser.add_argument('--mlp_test_split',  type=float, default=0.15)
    parser.add_argument('--skip_misclassified', action='store_true')
    parser.add_argument('--max_misclassified',  type=int, default=2000)
    parser.add_argument('--output_dir',      type=str,   default='results_plots')
    parser.add_argument('--skip_visualization',  action='store_true')
    parser.add_argument('--skip_classification', action='store_true')
    parser.add_argument('--wandb_project',   type=str,   default='autoencoder-egms-latent')
    parser.add_argument('--wandb_entity',    type=str,   default=None)
    parser.add_argument('--no_wandb',        action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    pth_paths = args.pth_path
    model_class_override = args.model_class

    if args.dataset_path is None and args.data_dir is None:
        print("ERROR: Provide either --data_dir or --dataset_path.")
        sys.exit(1)

    for pth_path in pth_paths:
        if not os.path.isfile(pth_path):
            print(f"\nWARNING: .pth not found, skipping: {pth_path}")
            continue

        pth_stem = os.path.splitext(os.path.basename(pth_path))[0]

        if model_class_override is not None:
            model_class = model_class_override
        else:
            try:
                model_class = _arch_from_stem(pth_stem)
            except ValueError as e:
                print(f"\nWARNING: {e}\nSkipping {pth_path}.")
                continue

        if args.dataset_path is not None:
            dataset_path = args.dataset_path
        else:
            try:
                signal_type  = _signal_type_from_stem(pth_stem)
                dataset_path = _find_dataset_path(args.data_dir, signal_type)
                print(f"  [auto-detect] signal_type={signal_type} -> {dataset_path}")
            except (ValueError, FileNotFoundError) as e:
                print(f"\nWARNING: {e}\nSkipping {pth_path}.")
                continue

        args.pth_path     = pth_path
        args.model_class  = model_class
        args.dataset_path = dataset_path

        if not args.no_wandb and WANDB_AVAILABLE:
            wandb.init(
                project = args.wandb_project,
                entity  = args.wandb_entity,
                name    = pth_stem,
                config  = vars(args),
                reinit  = True,
            )

        if not args.skip_visualization:
            visualize_latent_space(args)

        if not args.skip_classification:
            classify_latent_space(args)

        if not args.no_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()

    print("\nDone.")


if __name__ == '__main__':
    main()
