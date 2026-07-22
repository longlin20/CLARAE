"""
plot_tsne_patients.py
=====================
t-SNE del espacio latente por paciente, un subplot por ritmo.
Soporta train.h5, val.h5 y test.h5 (individualmente o en combinación).

Uso:
    python plot_tsne_patients.py \\
        --pth_path model_pth/bipolar/BIPOLAR_CLARAE_SCM_GLU_AP_ELU_dtw_noise.pth \\
        --dataset_path processed_data_final/bipolar_181/normalized/p0.05_99.95

    python plot_tsne_patients.py \\
        --pth_path model_pth/bipolar/BIPOLAR_CLARAE_SCM_GLU_AP_ELU_dtw_noise.pth \\
        --dataset_path processed_data_final/bipolar_181/normalized/p0.05_99.95 \\
        --splits train val test \\
        --n_signals_per_class 500 \\
        --wandb_project autoencoder-egms-tsne
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from sklearn.manifold import TSNE

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.colors as pc
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from evaluation.run_test import build_eval_args
from eval_utils import (
    load_model, extract_latent,
    _signal_type_from_stem, _find_dataset_path,
    _decode_str_array, EXCLUDED_RHYTHMS, RHYTHM_COLORS, load_split_h5,
)

# Marker style per split (used by the combined t-SNE figure)
SPLIT_MARKERS  = {'train': 'o',       'val': '^',           'test': 's'}
SPLIT_SIZES    = {'train': 8,         'val': 10,            'test': 14}
SPLIT_SYMBOLS_PLOTLY = {'train': 'circle', 'val': 'triangle-up', 'test': 'square'}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_h5(dataset_path: str, split: str):
    """Load signals, rhythms and patient_ids for a split.

    Delegates to load_split_h5 which supports chunked ({split}_001.h5 …)
    and legacy ({split}.h5) formats.
    """
    data = load_split_h5(dataset_path, split, include_signals=True)
    rhythms     = np.where(data['rhythms'] == 'RS', 'SR', data['rhythms'])
    mask        = np.array([r not in EXCLUDED_RHYTHMS for r in rhythms])
    signals     = data['signals'][mask].astype(np.float32)
    rhythms     = rhythms[mask]
    patient_ids = data['patient_ids'][mask]

    unique_r, counts_r = np.unique(rhythms, return_counts=True)
    print(f"  {len(rhythms):,} signals | "
          + ", ".join(f"{r}:{c:,}" for r, c in zip(unique_r, counts_r)))
    unique_p = np.unique(patient_ids)
    print(f"  {len(unique_p)} patients: {list(unique_p)}")
    return signals, rhythms, patient_ids


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _patient_colormap(unique_patients):
    """
    Assign a distinct color to each patient.
    Combines tab20 + tab20b + tab20c to cover up to 60 patients.
    """
    cmaps = [
        plt.cm.get_cmap('tab20',  20),
        plt.cm.get_cmap('tab20b', 20),
        plt.cm.get_cmap('tab20c', 20),
    ]
    colors = {}
    for i, pid in enumerate(unique_patients):
        cmap_idx  = i // 20
        color_idx = i % 20
        colors[pid] = cmaps[min(cmap_idx, 2)](color_idx)
    return colors


# ---------------------------------------------------------------------------
# Plotly interactive figures
# ---------------------------------------------------------------------------

def _patient_palette(unique_patients):
    """Assign a distinct plotly color to each patient."""
    palette = (pc.qualitative.Alphabet
               + pc.qualitative.Dark24
               + pc.qualitative.Light24)
    return {pid: palette[i % len(palette)]
            for i, pid in enumerate(unique_patients)}


def _build_plotly_figures(split, emb, rhythms, patient_ids,
                           unique_rhythms, unique_patients, pth_stem, elapsed):
    """
    Build two types of interactive Plotly figures:
      - fig_combined : subplots grid (one per rhythm), colored by patient
      - figs_per_rhy : dict {rhythm: go.Figure}  — one full figure per rhythm

    Points are interactive: hover shows patient ID.
    """
    pal = _patient_palette(unique_patients)
    n_rhy = len(unique_rhythms)
    ncols = 3
    nrows = (n_rhy + ncols - 1) // ncols

    subplot_titles = [
        f'{r}  (n={np.sum(rhythms == r):,})' for r in unique_rhythms
    ] + [''] * (nrows * ncols - n_rhy)

    fig_combined = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )

    shown_in_legend = set()
    for i, rhy in enumerate(unique_rhythms):
        row, col = i // ncols + 1, i % ncols + 1
        mask_rhy = rhythms == rhy
        emb_rhy  = emb[mask_rhy]
        pids_rhy = patient_ids[mask_rhy]

        # Gray background (other rhythms)
        fig_combined.add_trace(go.Scattergl(
            x=emb[~mask_rhy, 0], y=emb[~mask_rhy, 1],
            mode='markers',
            marker=dict(color='lightgray', size=2, opacity=0.15),
            showlegend=False, hoverinfo='skip',
        ), row=row, col=col)

        for pid in unique_patients:
            m = pids_rhy == pid
            if not m.any():
                continue
            show_leg = pid not in shown_in_legend
            shown_in_legend.add(pid)
            fig_combined.add_trace(go.Scattergl(
                x=emb_rhy[m, 0], y=emb_rhy[m, 1],
                mode='markers',
                marker=dict(color=pal[pid], size=4, opacity=0.75),
                name=pid,
                legendgroup=pid,
                showlegend=show_leg,
                hovertemplate=f'<b>{pid}</b><br>{rhy}<extra></extra>',
            ), row=row, col=col)

    fig_combined.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig_combined.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig_combined.update_layout(
        title=dict(
            text=(f't-SNE — espacio latente por paciente  [{split}]<br>'
                  f'<sup>{pth_stem}  |  {len(unique_patients)} pacientes  |  {elapsed:.0f}s</sup>'),
            font=dict(size=14),
        ),
        height=420 * nrows,
        width=520 * ncols,
        legend=dict(font=dict(size=9), itemsizing='constant'),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )

    # ── Per-rhythm figures ────────────────────────────────────────────────────
    figs_per_rhy = {}
    for rhy in unique_rhythms:
        mask_rhy     = rhythms == rhy
        emb_rhy      = emb[mask_rhy]
        pids_rhy     = patient_ids[mask_rhy]
        pids_present = [p for p in unique_patients if (pids_rhy == p).any()]

        fig_r = go.Figure()
        fig_r.add_trace(go.Scattergl(
            x=emb[~mask_rhy, 0], y=emb[~mask_rhy, 1],
            mode='markers',
            marker=dict(color='lightgray', size=3, opacity=0.15),
            showlegend=False, hoverinfo='skip',
        ))
        for pid in pids_present:
            m = pids_rhy == pid
            fig_r.add_trace(go.Scattergl(
                x=emb_rhy[m, 0], y=emb_rhy[m, 1],
                mode='markers',
                marker=dict(color=pal[pid], size=5, opacity=0.8),
                name=pid,
                hovertemplate=f'<b>{pid}</b><extra></extra>',
            ))
        fig_r.update_layout(
            title=dict(
                text=(f't-SNE — {rhy}  [{split}]  '
                      f'(n={mask_rhy.sum():,}, {len(pids_present)} pacientes)<br>'
                      f'<sup>{pth_stem}</sup>'),
                font=dict(size=13),
            ),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            legend=dict(font=dict(size=9), itemsizing='constant'),
            height=650,
            paper_bgcolor='white',
            plot_bgcolor='white',
        )
        figs_per_rhy[rhy] = fig_r

    return fig_combined, figs_per_rhy


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------

def process_split(split, dataset_path, model, eval_args, args,
                  pth_stem, use_wandb):
    """Load one split, run t-SNE, save figures, log to wandb."""

    print(f"\n{'='*60}")
    print(f"  Split: {split}  ({dataset_path})")
    print(f"{'='*60}")

    signals, rhythms, patient_ids = _load_h5(dataset_path, split)

    unique_rhythms  = sorted(np.unique(rhythms))
    unique_patients = sorted(np.unique(patient_ids))
    n_patients      = len(unique_patients)

    # ── Submuestreo opcional ─────────────────────────────────────────────────
    if args.n_signals_per_class is not None:
        rng  = np.random.default_rng(args.seed)
        keep = []
        for rhy in unique_rhythms:
            idx = np.where(rhythms == rhy)[0]
            if len(idx) > args.n_signals_per_class:
                idx = rng.choice(idx, args.n_signals_per_class, replace=False)
            keep.extend(idx.tolist())
        keep = np.array(sorted(keep))
        signals, rhythms, patient_ids = (
            signals[keep], rhythms[keep], patient_ids[keep]
        )
        print(f"  Tras submuestreo: {len(signals):,} señales")

    # ── Extraer vectores latentes ────────────────────────────────────────────
    print(f"\nExtrayendo vectores latentes [{split}] ...")
    t0     = time.time()
    latent = extract_latent(model, signals, eval_args.model_architecture,
                            args.batch_size, args.device)
    print(f"  Latent shape: {latent.shape}  ({time.time()-t0:.1f}s)")

    # ── t-SNE ────────────────────────────────────────────────────────────────
    print(f"\nEjecutando t-SNE (perplexity={args.tsne_perplexity}, "
          f"n_iter={args.tsne_n_iter}) [{split}] ...")
    t0  = time.time()
    emb = TSNE(
        n_components = 2,
        perplexity   = args.tsne_perplexity,
        max_iter     = args.tsne_n_iter,
        init         = 'pca',
        random_state = args.seed,
    ).fit_transform(latent)
    elapsed = time.time() - t0
    print(f"  t-SNE listo en {elapsed:.1f}s")

    # ── Colores de pacientes ─────────────────────────────────────────────────
    patient_colors = _patient_colormap(unique_patients)

    # ── Figura combinada (todos los ritmos en subplots) ──────────────────────
    n_rhy  = len(unique_rhythms)
    ncols  = 3
    nrows  = (n_rhy + ncols - 1) // ncols

    fig_all, axes_all = plt.subplots(nrows, ncols,
                                     figsize=(7 * ncols, 6 * nrows),
                                     squeeze=False)
    axes_flat = axes_all.flatten()

    for i, rhy in enumerate(unique_rhythms):
        ax       = axes_flat[i]
        mask_rhy = rhythms == rhy

        ax.scatter(emb[~mask_rhy, 0], emb[~mask_rhy, 1],
                   c='lightgray', s=3, alpha=0.15, rasterized=True, zorder=1)

        emb_rhy      = emb[mask_rhy]
        pids_rhy     = patient_ids[mask_rhy]
        pids_present = [p for p in unique_patients if (pids_rhy == p).any()]
        for pid in pids_present:
            m = pids_rhy == pid
            ax.scatter(emb_rhy[m, 0], emb_rhy[m, 1],
                       c=[patient_colors[pid]], s=10, alpha=0.75,
                       rasterized=True, zorder=2)

        ax.set_title(f'{rhy}  (n={mask_rhy.sum():,}  pat={len(pids_present)})',
                     fontweight='bold', fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(True, linestyle='--', alpha=0.25)

    for j in range(n_rhy, len(axes_flat)):
        axes_flat[j].axis('off')

    legend_handles = [
        mpatches.Patch(color=patient_colors[pid], label=pid)
        for pid in unique_patients
    ]
    fig_all.legend(
        handles        = legend_handles,
        title          = 'Paciente',
        ncol           = max(1, n_patients // 5),
        loc            = 'lower center',
        bbox_to_anchor = (0.5, -0.01),
        fontsize       = 7,
        title_fontsize = 8,
        frameon        = True,
    )
    fig_all.suptitle(
        f't-SNE — Espacio latente por paciente  [{split}]\n'
        f'{pth_stem}   ({n_patients} pacientes, {elapsed:.0f}s)',
        fontweight='bold', fontsize=14,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if use_wandb:
        wandb.log({f'latent/{split}/tsne_patients_combined': wandb.Image(fig_all)})
    plt.close(fig_all)

    # ── Figura individual por ritmo ──────────────────────────────────────────
    for rhy in unique_rhythms:
        mask_rhy     = rhythms == rhy
        emb_rhy      = emb[mask_rhy]
        pids_rhy     = patient_ids[mask_rhy]
        pids_present = [p for p in unique_patients if (pids_rhy == p).any()]

        fig_r, ax_r = plt.subplots(figsize=(9, 7))

        ax_r.scatter(emb[~mask_rhy, 0], emb[~mask_rhy, 1],
                     c='lightgray', s=3, alpha=0.15, rasterized=True, zorder=1)

        for pid in pids_present:
            m = pids_rhy == pid
            ax_r.scatter(emb_rhy[m, 0], emb_rhy[m, 1],
                         c=[patient_colors[pid]], s=14, alpha=0.8,
                         label=pid, rasterized=True, zorder=2)

        ax_r.set_title(
            f't-SNE — {rhy}  [{split}]  '
            f'(n={mask_rhy.sum():,}, {len(pids_present)} pacientes)\n'
            f'{pth_stem}',
            fontweight='bold',
        )
        ax_r.set_xticks([]); ax_r.set_yticks([])
        ax_r.grid(True, linestyle='--', alpha=0.25)

        handles_r = [mpatches.Patch(color=patient_colors[p], label=p)
                     for p in pids_present]
        ax_r.legend(handles=handles_r, title='Paciente',
                    ncol=max(1, len(pids_present) // 8),
                    fontsize=7, title_fontsize=8,
                    loc='best', frameon=True)
        plt.tight_layout()

        if use_wandb:
            wandb.log({f'latent/{split}/tsne_patients_{rhy}': wandb.Image(fig_r)})
        plt.close(fig_r)

    # ── Plotly interactive figures ────────────────────────────────────────────
    if PLOTLY_AVAILABLE and use_wandb:
        fig_combined_px, figs_per_rhy_px = _build_plotly_figures(
            split, emb, rhythms, patient_ids,
            unique_rhythms, unique_patients, pth_stem, elapsed,
        )
        wandb.log({
            f'latent/{split}/tsne_plotly_combined': wandb.Plotly(fig_combined_px)
        })
        for rhy, fig_r_px in figs_per_rhy_px.items():
            wandb.log({
                f'latent/{split}/tsne_plotly_{rhy}': wandb.Plotly(fig_r_px)
            })


# ---------------------------------------------------------------------------
# Combined t-SNE (all splits in one embedding)
# ---------------------------------------------------------------------------

def process_splits_combined(splits, dataset_path, model, eval_args, args,
                             pth_stem, use_wandb):
    """
    Load all requested splits, run a single t-SNE on the concatenated latent
    space, and generate two figures:
      Fig 1 — subplots per rhythm, colored by patient, marker shape = split
      Fig 2 — single plot, colored by rhythm, marker shape = split
    """
    print(f"\n{'='*60}")
    print(f"  Combined t-SNE: {' + '.join(splits)}")
    print(f"{'='*60}")

    all_signals     = []
    all_rhythms     = []
    all_patient_ids = []
    all_splits      = []

    rng = np.random.default_rng(args.seed)

    for split in splits:
        print(f"\n  Loading split: {split}")
        signals, rhythms, patient_ids = _load_h5(dataset_path, split)

        if args.n_signals_per_class is not None:
            keep = []
            for rhy in np.unique(rhythms):
                idx = np.where(rhythms == rhy)[0]
                if len(idx) > args.n_signals_per_class:
                    idx = rng.choice(idx, args.n_signals_per_class, replace=False)
                keep.extend(idx.tolist())
            keep = np.array(sorted(keep))
            signals, rhythms, patient_ids = (
                signals[keep], rhythms[keep], patient_ids[keep]
            )
            print(f"    After subsampling: {len(signals):,} signals")

        all_signals.append(signals)
        all_rhythms.append(rhythms)
        all_patient_ids.append(patient_ids)
        all_splits.append(np.full(len(signals), split))

    all_signals     = np.concatenate(all_signals,     axis=0)
    all_rhythms     = np.concatenate(all_rhythms,     axis=0)
    all_patient_ids = np.concatenate(all_patient_ids, axis=0)
    all_splits      = np.concatenate(all_splits,      axis=0)

    print(f"\n  Total signals: {len(all_signals):,}")

    # ── Latent extraction ────────────────────────────────────────────────────
    print(f"\nExtracting latent vectors (all splits) ...")
    t0     = time.time()
    latent = extract_latent(model, all_signals, eval_args.model_architecture,
                            args.batch_size, args.device)
    print(f"  Latent shape: {latent.shape}  ({time.time()-t0:.1f}s)")

    # ── t-SNE ────────────────────────────────────────────────────────────────
    print(f"\nRunning t-SNE (perplexity={args.tsne_perplexity}, "
          f"n_iter={args.tsne_n_iter}) ...")
    t0  = time.time()
    emb = TSNE(
        n_components = 2,
        perplexity   = args.tsne_perplexity,
        max_iter     = args.tsne_n_iter,
        init         = 'pca',
        random_state = args.seed,
    ).fit_transform(latent)
    elapsed = time.time() - t0
    print(f"  t-SNE ready in {elapsed:.1f}s")

    unique_rhythms  = sorted(np.unique(all_rhythms))
    unique_patients = sorted(np.unique(all_patient_ids))
    patient_colors  = _patient_colormap(unique_patients)
    split_label     = '_'.join(splits)

    # ── Matplotlib Fig 1: per-rhythm subplots, color=patient, shape=split ────
    n_rhy = len(unique_rhythms)
    ncols = 3
    nrows = (n_rhy + ncols - 1) // ncols

    fig1, axes1 = plt.subplots(nrows, ncols,
                               figsize=(7 * ncols, 6 * nrows),
                               squeeze=False)
    axes_flat = axes1.flatten()

    for i, rhy in enumerate(unique_rhythms):
        ax       = axes_flat[i]
        mask_rhy = all_rhythms == rhy

        ax.scatter(emb[~mask_rhy, 0], emb[~mask_rhy, 1],
                   c='lightgray', s=3, alpha=0.15, rasterized=True, zorder=1)

        emb_rhy      = emb[mask_rhy]
        pids_rhy     = all_patient_ids[mask_rhy]
        splits_rhy   = all_splits[mask_rhy]
        pids_present = [p for p in unique_patients if (pids_rhy == p).any()]

        for pid in pids_present:
            m_pid = pids_rhy == pid
            for split in splits:
                m = m_pid & (splits_rhy == split)
                if not m.any():
                    continue
                ax.scatter(emb_rhy[m, 0], emb_rhy[m, 1],
                           c=[patient_colors[pid]],
                           marker=SPLIT_MARKERS[split],
                           s=SPLIT_SIZES[split],
                           alpha=0.75, rasterized=True, zorder=2)

        ax.set_title(f'{rhy}  (n={mask_rhy.sum():,}  pat={len(pids_present)})',
                     fontweight='bold', fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(True, linestyle='--', alpha=0.25)

    for j in range(n_rhy, len(axes_flat)):
        axes_flat[j].axis('off')

    legend_patients = [mpatches.Patch(color=patient_colors[p], label=p)
                       for p in unique_patients]
    legend_splits   = [plt.Line2D([0], [0], marker=SPLIT_MARKERS[s],
                                  color='gray', linestyle='None',
                                  markersize=7, label=s)
                       for s in splits]
    fig1.legend(
        handles        = legend_patients + legend_splits,
        title          = 'Patient / Split',
        ncol           = max(1, len(unique_patients) // 5),
        loc            = 'lower center',
        bbox_to_anchor = (0.5, -0.01),
        fontsize       = 7,
        title_fontsize = 8,
        frameon        = True,
    )
    fig1.suptitle(
        f't-SNE — Latent space by patient  [{split_label}]\n'
        f'{pth_stem}   ({len(unique_patients)} patients, {elapsed:.0f}s)',
        fontweight='bold', fontsize=14,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if use_wandb:
        wandb.log({'latent/combined/tsne_by_patient': wandb.Image(fig1)})
    plt.close(fig1)

    # ── Matplotlib Fig 2: single plot, color=rhythm, shape=split ─────────────
    fig2, ax2 = plt.subplots(figsize=(10, 8))

    for rhy in unique_rhythms:
        rhy_color = RHYTHM_COLORS.get(rhy, '#888888')
        mask_rhy  = all_rhythms == rhy
        for split in splits:
            m = mask_rhy & (all_splits == split)
            if not m.any():
                continue
            ax2.scatter(emb[m, 0], emb[m, 1],
                        c=rhy_color,
                        marker=SPLIT_MARKERS[split],
                        s=SPLIT_SIZES[split],
                        alpha=0.65, rasterized=True)

    ax2.set_title(
        f't-SNE — Latent space by rhythm  [{split_label}]\n{pth_stem}',
        fontweight='bold', fontsize=14,
    )
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.grid(True, linestyle='--', alpha=0.25)

    legend_rhy     = [mpatches.Patch(color=RHYTHM_COLORS.get(r, '#888888'), label=r)
                      for r in unique_rhythms]
    legend_splits2 = [plt.Line2D([0], [0], marker=SPLIT_MARKERS[s],
                                  color='gray', linestyle='None',
                                  markersize=7, label=s)
                      for s in splits]
    ax2.legend(handles=legend_rhy + legend_splits2,
               title='Rhythm / Split', ncol=2,
               fontsize=8, loc='best', frameon=True)
    plt.tight_layout()

    if use_wandb:
        wandb.log({'latent/combined/tsne_by_rhythm': wandb.Image(fig2)})
    plt.close(fig2)

    # ── Matplotlib Fig 3: one figure per rhythm, color=patient, shape=split ──
    for rhy in unique_rhythms:
        mask_rhy     = all_rhythms == rhy
        emb_rhy      = emb[mask_rhy]
        pids_rhy     = all_patient_ids[mask_rhy]
        splits_rhy   = all_splits[mask_rhy]
        pids_present = [p for p in unique_patients if (pids_rhy == p).any()]

        fig_r, ax_r = plt.subplots(figsize=(9, 7))

        ax_r.scatter(emb[~mask_rhy, 0], emb[~mask_rhy, 1],
                     c='lightgray', s=3, alpha=0.15, rasterized=True, zorder=1)

        for pid in pids_present:
            m_pid = pids_rhy == pid
            for split in splits:
                m = m_pid & (splits_rhy == split)
                if not m.any():
                    continue
                ax_r.scatter(emb_rhy[m, 0], emb_rhy[m, 1],
                             c=[patient_colors[pid]],
                             marker=SPLIT_MARKERS[split],
                             s=SPLIT_SIZES[split] * 2,
                             alpha=0.8, rasterized=True, zorder=2)

        ax_r.set_title(
            f't-SNE — {rhy}  [{split_label}]  '
            f'(n={mask_rhy.sum():,}, {len(pids_present)} patients)\n{pth_stem}',
            fontweight='bold',
        )
        ax_r.set_xticks([]); ax_r.set_yticks([])
        ax_r.grid(True, linestyle='--', alpha=0.25)

        leg_pat = [mpatches.Patch(color=patient_colors[p], label=p)
                   for p in pids_present]
        leg_spl = [plt.Line2D([0], [0], marker=SPLIT_MARKERS[s],
                               color='gray', linestyle='None',
                               markersize=7, label=s)
                   for s in splits]
        ax_r.legend(handles=leg_pat + leg_spl,
                    title='Patient / Split',
                    ncol=max(1, len(pids_present) // 8),
                    fontsize=7, title_fontsize=8,
                    loc='best', frameon=True)
        plt.tight_layout()

        if use_wandb:
            wandb.log({f'latent/combined/tsne_{rhy}_by_patient': wandb.Image(fig_r)})
        plt.close(fig_r)

    # ── Plotly Fig 1: per-rhythm subplots, color=patient, symbol=split ───────
    if PLOTLY_AVAILABLE and use_wandb:
        pal = _patient_palette(unique_patients)

        subplot_titles = [
            f'{r}  (n={np.sum(all_rhythms == r):,})' for r in unique_rhythms
        ] + [''] * (nrows * ncols - n_rhy)

        pfig1 = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.04,
            vertical_spacing=0.08,
        )
        shown_leg = set()
        for i, rhy in enumerate(unique_rhythms):
            row, col = i // ncols + 1, i % ncols + 1
            mask_rhy = all_rhythms == rhy

            pfig1.add_trace(go.Scattergl(
                x=emb[~mask_rhy, 0], y=emb[~mask_rhy, 1],
                mode='markers',
                marker=dict(color='lightgray', size=2, opacity=0.15),
                showlegend=False, hoverinfo='skip',
            ), row=row, col=col)

            emb_rhy    = emb[mask_rhy]
            pids_rhy   = all_patient_ids[mask_rhy]
            splits_rhy = all_splits[mask_rhy]

            for pid in unique_patients:
                m_pid = pids_rhy == pid
                if not m_pid.any():
                    continue
                for split in splits:
                    m = m_pid & (splits_rhy == split)
                    if not m.any():
                        continue
                    leg_key = (pid, split)
                    show_leg = leg_key not in shown_leg
                    shown_leg.add(leg_key)
                    pfig1.add_trace(go.Scattergl(
                        x=emb_rhy[m, 0], y=emb_rhy[m, 1],
                        mode='markers',
                        marker=dict(color=pal[pid], size=5,
                                    symbol=SPLIT_SYMBOLS_PLOTLY[split],
                                    opacity=0.75),
                        name=f'{pid} [{split}]',
                        legendgroup=f'{pid}_{split}',
                        showlegend=show_leg,
                        hovertemplate=f'<b>{pid}</b> [{split}]<br>{rhy}<extra></extra>',
                    ), row=row, col=col)

        pfig1.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        pfig1.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        pfig1.update_layout(
            title=dict(
                text=(f't-SNE — combined [{split_label}] — by patient<br>'
                      f'<sup>{pth_stem}  |  {len(unique_patients)} patients'
                      f'  |  {elapsed:.0f}s</sup>'),
                font=dict(size=14),
            ),
            height=420 * nrows,
            width=520 * ncols,
            legend=dict(font=dict(size=8), itemsizing='constant'),
            paper_bgcolor='white', plot_bgcolor='white',
        )
        wandb.log({'latent/combined/tsne_plotly_by_patient': wandb.Plotly(pfig1)})

        # ── Plotly Fig 2: single plot, color=rhythm, symbol=split ────────────
        pfig2 = go.Figure()
        for rhy in unique_rhythms:
            rhy_color = RHYTHM_COLORS.get(rhy, '#888888')
            mask_rhy  = all_rhythms == rhy
            for split in splits:
                m = mask_rhy & (all_splits == split)
                if not m.any():
                    continue
                pfig2.add_trace(go.Scattergl(
                    x=emb[m, 0], y=emb[m, 1],
                    mode='markers',
                    marker=dict(color=rhy_color, size=5,
                                symbol=SPLIT_SYMBOLS_PLOTLY[split],
                                opacity=0.65),
                    name=f'{rhy} [{split}]',
                    legendgroup=rhy,
                    hovertemplate=f'<b>{rhy}</b> [{split}]<extra></extra>',
                ))
        pfig2.update_layout(
            title=dict(
                text=(f't-SNE — combined [{split_label}] — by rhythm<br>'
                      f'<sup>{pth_stem}  |  {elapsed:.0f}s</sup>'),
                font=dict(size=14),
            ),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            legend=dict(font=dict(size=9), itemsizing='constant'),
            height=700,
            paper_bgcolor='white', plot_bgcolor='white',
        )
        wandb.log({'latent/combined/tsne_plotly_by_rhythm': wandb.Plotly(pfig2)})

        # ── Plotly Fig 3: one figure per rhythm, color=patient, symbol=split ─
        for rhy in unique_rhythms:
            mask_rhy     = all_rhythms == rhy
            emb_rhy      = emb[mask_rhy]
            pids_rhy     = all_patient_ids[mask_rhy]
            splits_rhy   = all_splits[mask_rhy]
            pids_present = [p for p in unique_patients if (pids_rhy == p).any()]

            fig_rp = go.Figure()
            fig_rp.add_trace(go.Scattergl(
                x=emb[~mask_rhy, 0], y=emb[~mask_rhy, 1],
                mode='markers',
                marker=dict(color='lightgray', size=3, opacity=0.15),
                showlegend=False, hoverinfo='skip',
            ))
            for pid in pids_present:
                m_pid = pids_rhy == pid
                for split in splits:
                    m = m_pid & (splits_rhy == split)
                    if not m.any():
                        continue
                    fig_rp.add_trace(go.Scattergl(
                        x=emb_rhy[m, 0], y=emb_rhy[m, 1],
                        mode='markers',
                        marker=dict(color=pal[pid], size=6,
                                    symbol=SPLIT_SYMBOLS_PLOTLY[split],
                                    opacity=0.8),
                        name=f'{pid} [{split}]',
                        legendgroup=f'{pid}_{split}',
                        hovertemplate=f'<b>{pid}</b> [{split}]<extra></extra>',
                    ))
            fig_rp.update_layout(
                title=dict(
                    text=(f't-SNE — {rhy}  [{split_label}]  '
                          f'(n={mask_rhy.sum():,}, {len(pids_present)} patients)<br>'
                          f'<sup>{pth_stem}</sup>'),
                    font=dict(size=13),
                ),
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                legend=dict(font=dict(size=9), itemsizing='constant'),
                height=650,
                paper_bgcolor='white', plot_bgcolor='white',
            )
            wandb.log({f'latent/combined/tsne_plotly_{rhy}': wandb.Plotly(fig_rp)})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='t-SNE del espacio latente por paciente, un subplot por ritmo.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--pth_path',     required=True,
                        help='Ruta al checkpoint .pth del autoencoder.')
    parser.add_argument('--data_dir',     default=None,
                        help='Directorio raíz de datos. Tipo de señal inferido del stem.')
    parser.add_argument('--dataset_path', default=None,
                        help='Directorio HDF5 explícito (sobreescribe --data_dir).')
    parser.add_argument('--splits',       nargs='+', default=['test'],
                        choices=['train', 'val', 'test'],
                        help='Splits a procesar (pueden ser varios).')
    parser.add_argument('--output_dir',   default='results_tsne',
                        help='Directory for saving combined t-SNE PNG files.')
    parser.add_argument('--combined_tsne', action='store_true',
                        help='Run a single t-SNE on all splits combined, '
                             'distinguishing splits by marker shape '
                             '(train=o, val=^, test=s).')
    parser.add_argument('--skip_per_split', action='store_true',
                        help='Skip the individual per-split t-SNE runs '
                             '(useful when only --combined_tsne is needed).')
    parser.add_argument('--tsne_perplexity', type=int,   default=200)
    parser.add_argument('--tsne_n_iter',     type=int,   default=1000)
    parser.add_argument('--n_signals_per_class', type=int, default=None,
                        help='Submuestreo por clase para acelerar el t-SNE (None = todos).')
    parser.add_argument('--latent_dim',      type=int,   default=64)
    parser.add_argument('--filters_initial', type=int,   default=64)
    parser.add_argument('--dense_dim',       type=int,   default=128)
    parser.add_argument('--batch_size',      type=int,   default=512)
    parser.add_argument('--seed',            type=int,   default=42)
    parser.add_argument('--device',          type=str,   default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--wandb_project',   type=str,   default=None)
    parser.add_argument('--wandb_entity',    type=str,   default=None)
    args = parser.parse_args()

    args.device = 'cuda' if (args.device == 'cuda'
                             and torch.cuda.is_available()) else 'cpu'
    print(f"Device: {args.device}")

    pth_stem = os.path.splitext(os.path.basename(args.pth_path))[0]

    # ── Resolver dataset_path ────────────────────────────────────────────────
    if args.dataset_path is None:
        if args.data_dir is None:
            raise ValueError("Proporciona --data_dir o --dataset_path.")
        signal_type       = _signal_type_from_stem(pth_stem)
        args.dataset_path = _find_dataset_path(args.data_dir, signal_type)
    print(f"Dataset : {args.dataset_path}")

    # ── Cargar modelo (una sola vez para todos los splits) ───────────────────
    print(f"\nCargando AE: {args.pth_path}")

    # Necesitamos el input_length — leemos los metadatos del primer split disponible
    first_split  = args.splits[0]
    first_data   = load_split_h5(args.dataset_path, first_split, include_signals=True)
    input_length = first_data['signals'].shape[2]
    del first_data

    eval_args = build_eval_args(pth_stem, argparse.Namespace(
        loss_function=None, device=args.device,
        wandb_project=None, wandb_entity=None, no_wandb=True,
    ))
    eval_args.latent_dim      = args.latent_dim
    eval_args.filters_initial = args.filters_initial
    eval_args.dense_dim       = args.dense_dim

    model = load_model(
        pth_path        = args.pth_path,
        model_class     = eval_args.model_architecture,
        latent_dim      = eval_args.latent_dim,
        filters_initial = eval_args.filters_initial,
        dropout_rate    = eval_args.dropout_rate,
        dense_dim       = eval_args.dense_dim,
        input_length    = input_length,
        device          = args.device,
        q_parameter     = eval_args.q_parameter,
    )

    # ── WandB init ───────────────────────────────────────────────────────────
    use_wandb = (WANDB_AVAILABLE and args.wandb_project is not None)
    if use_wandb:
        wandb.init(
            project = args.wandb_project,
            entity  = args.wandb_entity,
            name    = pth_stem + '_tsne_patients',
            reinit  = True,
        )

    # ── Procesar cada split ──────────────────────────────────────────────────
    if not args.skip_per_split:
        for split in args.splits:
            process_split(split, args.dataset_path, model, eval_args, args,
                          pth_stem, use_wandb)

    # ── Combined t-SNE (all splits in one embedding) ─────────────────────────
    if args.combined_tsne:
        process_splits_combined(args.splits, args.dataset_path, model,
                                eval_args, args, pth_stem, use_wandb)

    if use_wandb:
        wandb.finish()

    print("\nDone.")


if __name__ == '__main__':
    main()
