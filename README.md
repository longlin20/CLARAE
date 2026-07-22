# CLARAE: Convolutional Autoencoder for EGM Reconstruction and Enhancement

Official implementation of **[Paper Title]**, submitted to **[Venue, Year]**.

---

## Installation

```bash
git clone https://github.com/longlin20/CLARAE.git
cd CLARAE
pip install torch --index-url https://download.pytorch.org/whl/cu121  # adjust CUDA version
pip install -r requirements.txt
```

---

## Data Preparation

The model expects preprocessed EGM data in HDF5 format, split into `train`, `val`, and `test` sets.

**Directory structure:**
```
processed_data/
└── bipolar_N/
    └── normalized/
        └── p0.05_99.95/
            ├── train_001.h5
            ├── train_002.h5
            ├── val_001.h5
            └── test_001.h5
```

**Each `.h5` file must contain:**
- `{split}_data` — signal array of shape `(N, 1, L)`, normalized to `[-1, 1]`
- `{split}_rhythms` — string array of rhythm labels per signal
- `{split}_patient_ids` — patient identifier per signal
- Attributes `p_inf_value` and `p_sup_value` — percentile values used for normalization

Normalization formula:
```
normalized = 2 * (x - p_inf) / (p_sup - p_inf) - 1
```
where `p_inf` and `p_sup` are the 0.05th and 99.95th percentiles computed over the training set.

---

## Training CLARAE

```bash
python run_training.py \
    --preprocessed_data_dir processed_data/ \
    --model_architecture CLARAE_SCM_GLU_AP_ELU_GATE_SKLAST \
    --bipolar \
    --loss_function dtw \
    --add_noise \
    --random_noise \
    --epochs 150 \
    --latent_dim 64 \
    --filters_initial 64 \
    --dense_dim 64 \
    --no_wandb
```

**Key arguments:**
| Argument | Description | Default |
|---|---|---|
| `--model_architecture` | Architecture name (see `model_registry.py`) | — |
| `--bipolar` / `--unipolar` | Signal type | unipolar |
| `--loss_function` | `mse` or `dtw` | `mse` |
| `--add_noise` | Enable noise augmentation during training | off |
| `--random_noise` | Randomly mix 1–4 noise types per batch | off |
| `--noise_snr_min/max` | SNR range in dB | `-5.0` / `10.0` |
| `--epochs` | Number of training epochs | `100` |
| `--latent_dim` | Latent space dimension | `64` |
| `--no_wandb` | Disable Weights & Biases logging | off |

To run all model variants for comparison:
```bash
python run_all_models.py --bipolar --data_dir processed_data/ --loss_functions dtw --noise_modes noise
```

---

## Evaluation

**Reconstruction metrics** (R², ΔV_pp, ΔDF, NLEO-corr):
```bash
python evaluation/run_test.py \
    --pth_path model_pth/BIPOLAR_CLARAE_SCM_GLU_AP_ELU_GATE_SKLAST_dtw_noise.pth \
    --preprocessed_data_dir processed_data/ \
    --no_wandb
```

**External validation** on the [PhysioNet IAF database](https://physionet.org/content/iafdb/1.0.0/):
```bash
# 1. Build the IAF dataset
python iaf_database/build_iaf_dataset.py \
    --db_dir /path/to/intracardiac-atrial-fibrillation-database-1.0.0 \
    --out_h5 iaf_eval_output/iaf.h5

# 2. Evaluate
python iaf_database/eval_iaf_database.py \
    --pth_path model_pth/BIPOLAR_CLARAE_SCM_GLU_AP_ELU_GATE_SKLAST_dtw_noise.pth \
    --iaf_h5 iaf_eval_output/iaf.h5
```

---

## Repository Structure

```
├── architecture/          # Model architectures
│   ├── autoencoders_improved.py   # CLARAE (proposed)
│   ├── drnn.py / deepfilter.py    # Baselines
│   ├── dae.py                     # FCN-DAE / CNN-DAE
│   ├── acdae.py / fgdae.py        # Baselines
├── loss/
│   └── soft_dtw_loss.py           # Soft-DTW loss
├── metrics/               # Clinical metrics (R², Vpp, DF, NLEO, LATs)
├── functions/             # Signal processing utilities
├── plots/                 # Visualization scripts
├── evaluation/            # Evaluation scripts
│   ├── run_test.py                # Reconstruction metrics (R², Vpp, DF, NLEO)
│   ├── evaluate_clf.py            # Downstream rhythm classification
│   └── latent_space_analysis.py   # t-SNE and latent space visualization
├── iaf_database/          # External validation (PhysioNet IAF)
│   ├── build_iaf_dataset.py
│   └── eval_iaf_database.py
├── run_training.py        # Train a model
├── run_all_models.py      # Train all architectures for comparison
├── training_engine.py     # Train/validation loop
├── model_registry.py      # Central model registry
├── eval_utils.py          # Shared evaluation utilities
└── utils.py               # Noise generators, normalization
```

---

## Citation

```bibtex
@inproceedings{clarae2025,
  title  = {[Paper Title]},
  author = {[Authors]},
  year   = {2025}
}
```
