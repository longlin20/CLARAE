import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import device
from torch.utils.data import DataLoader, TensorDataset
import torch.optim.lr_scheduler as lr_scheduler
# from lion_pytorch import Lion
from glob import glob
import wandb
from eval_utils import load_split_h5

from training_engine import train_epoch, validate
# from loss.dilate_loss import DILATELoss  # Commented out - DILATE loss disabled
from loss.soft_dtw_loss import SoftDTWLoss
from model_registry import ALL_MODEL_NAMES, _CLARAE_MODELS, SKIP_DROP_MODELS, GATE_MODELS, get_model_class

from plots.plot_AE_results import visualize_reconstructions
from utils import EarlyStopping

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train autoencoder for EGM signal denoising')

    # Data arguments
    parser.add_argument('--preprocessed_data_dir', type=str, required=True,
                        help='Directory with preprocessed data (REQUIRED)')

    # Model arguments
    parser.add_argument('--model_architecture', type=str, default='CLARAE',
                        choices=ALL_MODEL_NAMES,
                        help='Model architecture to use')
    parser.add_argument('--unipolar', action='store_true', default=True,
                        help='Use unipolar signals (default: True)')
    parser.add_argument('--bipolar', dest='unipolar', action='store_false',
                        help='Use bipolar signals')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Latent dimension size')
    parser.add_argument('--filters_initial', type=int, default=128,
                        help='Initial number of filters')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--dense_dim', type=int, default=256,
                        help='Number of neurons in the FC dense layer (encoder/decoder) for CLARAE_SC, SCD, SCD2')

    # CLARAE SC models: skip connection dropout
    parser.add_argument('--skip_dropout', type=float, default=0.1,
                        help='Dropout rate for skip connections in CLARAE SC models (default: 0.1)')
    parser.add_argument('--gate_init', type=float, default=-5.0,
                        help='Initial logit value for skip gates; more negative = slower opening (default: -5.0)')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['Adam', 'RMSProp', 'SGD'],  # 'Lion' removed
                        help='Optimizer to use')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay for optimizer')

    # Lion optimizer specific arguments (commented out - Lion optimizer disabled)
    # parser.add_argument('--lion_beta1', type=float, default=0.9,
    #                     help='Beta1 for Lion optimizer (default: 0.9, use 0.95 for more stability)')
    # parser.add_argument('--lion_beta2', type=float, default=0.99,
    #                     help='Beta2 for Lion optimizer (default: 0.99, use 0.98 for more stability)')

    # Loss function arguments
    parser.add_argument('--loss_function', type=str, default='mse',
                        choices=['mse', 'dtw'],
                        help='Loss function to use (mse or dtw)')
    parser.add_argument('--dtw_gamma', type=float, default=1.0,
                        help='Gamma parameter for Soft-DTW (smoothing factor, default: 1.0)')

    # DILATE Loss arguments (commented out - DILATE loss disabled)
    # parser.add_argument('--dilate_alpha', type=float, default=0.5,
    #                     help='Alpha parameter for DILATE loss (0=temporal only, 1=shape only, 0.5=balanced, default: 0.5)')
    # parser.add_argument('--dilate_gamma', type=float, default=0.01,
    #                     help='Gamma parameter for DILATE loss (smoothing factor, default: 0.01)')


# Learning rate scheduler arguments
    parser.add_argument('--patience_lr', type=int, default=3,
                        help='Patience for learning rate reduction')
    parser.add_argument('--factor_lr', type=float, default=0.5,
                        help='Factor for learning rate reduction')
    parser.add_argument('--min_lr', type=float, default=1e-8,
                        help='Minimum learning rate')

    # Early stopping arguments
    parser.add_argument('--patience_es', type=int, default=7,
                        help='Patience for early stopping')
    parser.add_argument('--min_delta', type=float, default=1e-6,
                        help='Minimum delta for early stopping')

    # Preprocessing arguments
    parser.add_argument('--percentile_inf', type=float, default=0.05,
                        help='Lower percentile for normalization')
    parser.add_argument('--percentile_sup', type=float, default=99.95,
                        help='Upper percentile for normalization')

    # Model-specific arguments
    parser.add_argument('--q_parameter', type=int, default=2,
                        help='Q parameter for FGDAE model')
    parser.add_argument('--kl_beta', type=float, default=0.0,
                        help='Weight for KL divergence loss in VAE models (default: 0, disabled)')

    # Noise augmentation arguments
    parser.add_argument('--add_noise', action='store_true',
                        help='Add noise to training data for denoising task')
    parser.add_argument('--random_noise', action='store_true',
                        help='Randomly select 1-4 noise types per batch (overrides --noise_types)')
    parser.add_argument('--noise_types', type=str, nargs='+',
                        default=['gaussian'],
                        choices=['gaussian', 'baseline_wander', 'powerline', 'spike'],
                        help='Types of noise to add. Options: gaussian (thermal/electronic), '
                             'baseline_wander (low-freq drift, 1-4 sinusoids at 0.01-0.3Hz), '
                             'powerline (50/60Hz interference), spike (transient artifacts at 4-2400ms). '
                             'Can specify multiple: --noise_types gaussian powerline spike')
    parser.add_argument('--min_noise_types', type=int, default=1,
                        help='Minimum number of noise types when using --random_noise (default: 1)')
    parser.add_argument('--max_noise_types', type=int, default=4,
                        help='Maximum number of noise types when using --random_noise (default: 4)')
    parser.add_argument('--noise_snr_min', type=float, default=-5.0,
                        help='Minimum Signal-to-Noise Ratio in dB (default: -5.0, more noise)')
    parser.add_argument('--noise_snr_max', type=float, default=10.0,
                        help='Maximum Signal-to-Noise Ratio in dB (default: 10.0, less noise)')
    parser.add_argument('--powerline_freq', type=int, default=50,
                        choices=[50, 60],
                        help='Powerline frequency in Hz (50 for Europe, 60 for Americas, default: 50)')
    parser.add_argument('--sampling_freq', type=int, default=500,
                        help='Sampling frequency of the signals in Hz (default: 500)')

    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='autoencoder-egms-new',
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity (username or team)')
    parser.add_argument('--wandb_resume', type=str, default=None,
                        help='Wandb run ID to resume from')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')

    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--model_save_dir', type=str,
                        default=os.path.join(_SCRIPT_DIR, 'model_pth', 'new'),
                        help='Directory to save trained models')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Custom model name (default: auto-generated from architecture and signal type)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--noise_seed', type=int, default=0,
                        help='Fixed seed for noise generation during test evaluation (default: 0). '
                             'Ensures all models are evaluated with identical noise.')

    # DF quality filter
    parser.add_argument('--min_power', type=float, default=0.0,
                        help='Excluye señales con sum_power(DF) < min_power de train y val. '
                             '0.0 = sin filtro. Añade sufijo _pwXX al nombre del modelo.')

    # Debug/Testing arguments
    parser.add_argument('--debug_samples', type=int, default=None,
                        help='Load only N samples per split for quick testing (default: None = load all data)')

    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def print_arguments(args):
    """Print all arguments at the start of training"""
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)

    # Group arguments by category
    categories = {
        "Data Configuration": ['preprocessed_data_dir', 'percentile_inf', 'percentile_sup', 'debug_samples'],
        "Model Configuration": ['model_architecture', 'unipolar', 'latent_dim', 'filters_initial', 'dropout_rate', 'skip_dropout'],
        "Training Configuration": ['batch_size', 'epochs', 'learning_rate', 'optimizer', 'weight_decay', 'loss_function'],
        "Loss Function Parameters": ['dtw_gamma', 'slope_alpha', 'slope_m'],  # dilate_alpha, dilate_gamma removed
        "Noise Augmentation": ['add_noise', 'random_noise', 'noise_types', 'min_noise_types', 'max_noise_types', 'noise_snr_min', 'noise_snr_max', 'powerline_freq', 'sampling_freq', 'noise_seed'],
        "Learning Rate Scheduler": ['patience_lr', 'factor_lr', 'min_lr'],
        "Early Stopping": ['patience_es', 'min_delta'],
        "Wandb Configuration": ['wandb_project', 'wandb_entity', 'wandb_resume', 'no_wandb'],
        "Other": ['device', 'model_save_dir', 'model_name', 'seed']
    }

    for category, params in categories.items():
        print(f"\n{category}:")
        print("-" * 80)
        for param in params:
            value = getattr(args, param)
            print(f"  {param:25s}: {value}")

    print("\n" + "="*80 + "\n")

def print_model_architecture(model, args):
    """Print detailed model architecture with parameter counts"""
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    print(f"Model: {args.model_architecture}")
    print(f"Signal Type: {'UNIPOLAR' if args.unipolar else 'BIPOLAR'}")
    print("-" * 80)

    # Print encoder architecture if available
    if hasattr(model, 'encoder'):
        print("\nENCODER:")
        print("-" * 80)
        for name, module in model.encoder.named_children():
            n_params = sum(p.numel() for p in module.parameters())
            n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"  {name:20s}: {str(module):50s} | Params: {n_params:,} (trainable: {n_trainable:,})")

        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        print(f"\n  Total Encoder Params: {encoder_params / 1e6:.2f} M (trainable: {encoder_trainable / 1e6:.2f} M)")

    # Print decoder architecture if available
    if hasattr(model, 'decoder'):
        print("\nDECODER:")
        print("-" * 80)
        for name, module in model.decoder.named_children():
            n_params = sum(p.numel() for p in module.parameters())
            n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"  {name:20s}: {str(module):50s} | Params: {n_params:,} (trainable: {n_trainable:,})")

        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        decoder_trainable = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
        print(f"\n  Total Decoder Params: {decoder_params / 1e6:.2f} M (trainable: {decoder_trainable / 1e6:.2f} M)")

    # Print total parameters
    print("\nTOTAL MODEL:")
    print("-" * 80)
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_fixed = total_params - total_trainable

    print(f"  Total learnable params: {total_trainable / 1e6:.2f} M")
    print(f"  Total fixed params: {total_fixed / 1e6:.2f} M")
    print(f"  Total params: {total_params / 1e6:.2f} M")

    if wandb.run is not None:
        wandb.run.summary["total_params"] = total_params

    print("\n" + "="*80 + "\n")

def _filter_by_power(data: torch.Tensor, split_name: str, args,
                     batch_size: int = 2000) -> torch.Tensor:
    """Elimina señales con sum_power < args.min_power usando DF Welch.

    Procesa en mini-lotes para evitar acumular todos los espectros en memoria.
    """
    from metrics.DF import calculate_df_batch
    from tqdm import tqdm
    _DF_LOW, _DF_HIGH, _NFFT, _WIN_SEC = 0.8, 15.0, 4096, 2.5
    idx_low  = int(round(_DF_LOW  * _NFFT / args.sampling_freq))
    idx_high = int(round(_DF_HIGH * _NFFT / args.sampling_freq))

    n = len(data)
    sum_pw = np.empty(n, dtype=np.float32)

    for start in tqdm(range(0, n, batch_size),
                      desc=f"  DF filter [{split_name}]", unit="batch"):
        end      = min(start + batch_size, n)
        batch_np = data[start:end, 0, :].numpy()
        df_result = calculate_df_batch(
            batch_np, fs=args.sampling_freq,
            verbose=False, show_progress=False,
            minDF=_DF_LOW, maxDF=_DF_HIGH, Nfft=_NFFT,
            apply_filters=True, window_width_seconds=_WIN_SEC,
            window_overlapping=0.0,
        )
        spectra = np.array(df_result.DF_Spectrum)          # (batch, freq_bins)
        sum_pw[start:end] = spectra[:, idx_low:idx_high + 1].sum(axis=1)
        del df_result, spectra, batch_np                   # liberar memoria

    mask  = torch.from_numpy(sum_pw >= args.min_power)
    n_rm  = int((~mask).sum())
    print(f"  [{split_name}] power filter: removed {n_rm:,}/{n:,} "
          f"({100.0*n_rm/max(n,1):.1f}%)  remaining={int(mask.sum()):,}")
    return data[mask]


def load_preprocessed_data(args):
    """
    Load preprocessed data from split HDF5 files created by preprocess_database.py.
    Supports both chunked format (train_001.h5, …) and legacy (train.h5).

    Returns: (train_data, val_data, None, percentiles)
    """
    signal_type = 'unipolar' if args.unipolar else 'bipolar'
    percentile_name = f"p{args.percentile_inf}_{args.percentile_sup}"

    pattern = os.path.join(args.preprocessed_data_dir, f"{signal_type}_*")
    matching_dirs = glob(pattern)

    if len(matching_dirs) == 0:
        print(f"\n[ERROR] No preprocessed data directory found matching: {pattern}")
        print(f"Please run: python preprocess_database.py --signal_type {signal_type}")
        return None, None, None, None

    if len(matching_dirs) > 1:
        print(f"\n[WARNING] Multiple directories found matching: {pattern}")
        for d in matching_dirs:
            print(f"    - {d}")
        print(f"Using: {matching_dirs[0]}")

    data_dir = os.path.join(matching_dirs[0], 'normalized', percentile_name)
    print(f"\nLoading preprocessed data from: {data_dir}")

    n = args.debug_samples
    if n is not None:
        print(f"\n[DEBUG] DEBUG MODE: Loading only {n} samples per split")

    def _load(split_name):
        d = load_split_h5(data_dir, split_name, include_signals=True)
        signals = d['signals'][:n] if n is not None else d['signals']
        return torch.tensor(signals, dtype=torch.float32), d['p_inf'], d['p_sup']

    train_data, p_inf, p_sup = _load('train')
    val_data, _, _           = _load('val')
    percentiles = np.array([p_inf, p_sup])

    total_mb = sum(
        os.path.getsize(os.path.join(data_dir, fn)) / (1024 ** 2)
        for fn in os.listdir(data_dir)
        if fn.endswith('.h5') and not fn.startswith('clf_')
    )
    print(f"  File size: {total_mb:.2f} MB total")
    print(f"  Train: {train_data.shape}")
    print(f"  Val:   {val_data.shape}")
    print(f"  Percentiles: {percentiles}")

    return train_data, val_data, None, percentiles

def compile_model(args, input_channels, input_length):
    """Compile model, criterion, and optimizer based on arguments"""
    model_class = get_model_class(args.model_architecture)

    if args.model_architecture in _CLARAE_MODELS:
        kwargs = dict(
            input_channels=input_channels,
            input_length=input_length,
            latent_dim=args.latent_dim,
            filters_initial=args.filters_initial,
            dropout_rate=args.dropout_rate,
            dense_dim=args.dense_dim,
        )
        if args.model_architecture in SKIP_DROP_MODELS:
            kwargs['skip_dropout'] = args.skip_dropout
        if args.model_architecture in GATE_MODELS:
            kwargs['gate_init'] = args.gate_init
        model = model_class(**kwargs)
    elif args.model_architecture == "DRNN":
        model = model_class(
            input_length=input_length,
            hidden_size=args.latent_dim,
            dropout_rate=args.dropout_rate,
        )
    elif args.model_architecture in ["CNN-DAE", "FCN-DAE"]:
        model = model_class(input_length=input_length)
    elif args.model_architecture == "DEEP-FILTER":
        model = model_class(input_channels=input_channels)
    elif args.model_architecture == "ACDAE":
        model = model_class(
            input_channels=input_channels,
            signal_size=input_length,
        )
    elif args.model_architecture == "FGDAE":
        model = model_class(
            signal_size=input_length,
            q=args.q_parameter
        )

    # Select loss function
    if args.loss_function == 'mse':
        criterion = nn.MSELoss()
        print("Using loss function: MSE (Mean Squared Error)")

    elif args.loss_function == 'dtw':
        # pysdtw es mucho más simple y rápido que tslearn
        # No necesita wrapper complicado ni loop manual
        # NOTE: use_cuda=False due to numba/CUDA compatibility issues
        # CPU version is still 10-50x faster than tslearn
        criterion = SoftDTWLoss(
            gamma=args.dtw_gamma,
            use_cuda=False
        )
        print(f"Using loss function: Soft-DTW (gamma={args.dtw_gamma})")
        print(f"  Using CPU acceleration (CUDA disabled due to numba compatibility)")

    # elif args.loss_function == 'dilate':
    #     # DILATE: Combines shape (soft-DTW) and temporal (path alignment) losses
    #     # Reference: Le Guen & Thome (2019) NeurIPS
    #     # This loss returns a tuple: (total_loss, shape_loss, temporal_loss)
    #     use_cuda = device.type == 'cuda'
    #     criterion = DILATELoss(
    #         alpha=args.dilate_alpha,  # 0=temporal only, 1=shape only, 0.5=balanced
    #         gamma=args.dilate_gamma   # smoothing parameter
    #     )
    #     print(f"Using loss function: DILATE (Shape + Temporal Distortion)")
    #     print(f"  Alpha (shape vs temporal): {args.dilate_alpha}")
    #     print(f"  Gamma (smoothing): {args.dilate_gamma}")
    #     print(f"  Reference: Le Guen & Thome (2019) 'Shape and Time Distortion Loss' - NeurIPS 2019")

    else:
        raise ValueError(f"Unknown loss function: {args.loss_function}")

    # Select optimizer
    optimizers = {
        "Adam": optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay),
        "RMSProp": optim.RMSprop(model.parameters(), lr=args.learning_rate),
        "SGD": optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9),
        # "Lion": Lion(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(args.lion_beta1, args.lion_beta2), use_triton=True),  # Lion optimizer disabled
    }

    optimizer = optimizers[args.optimizer]

    return model, criterion, optimizer

def main():
    # Parse arguments
    print(f"Begin\n")
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

    # Initialize wandb
    model_type = "UNIPOLAR" if args.unipolar else "BIPOLAR"

    # Use custom model name if provided, otherwise generate automatically
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = f"{args.model_architecture}_{model_type}_v4"
        if args.min_power > 0.0:
            model_name += f"_pw{int(args.min_power * 100):02d}"

    print(f"Model name: {model_name}\n")

    if not args.no_wandb:
        # Check if resuming a run
        if args.wandb_resume:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                id=args.wandb_resume,
                resume="must"
            )
            print(f"Resuming wandb run: {args.wandb_resume}\n")
        else:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=model_name,
                config=vars(args)
            )
            print(f"Started new wandb run: {wandb.run.id}\n")

    # Print all arguments after wandb initialization
    print_arguments(args)

    # Create model save directory
    os.makedirs(args.model_save_dir, exist_ok=True)
    model_path = os.path.join(args.model_save_dir, f"{model_name}.pth")

    # Check if model already exists
    if os.path.exists(model_path) and not args.wandb_resume:
        print(f"Model already exists at {model_path}")
        print("Use --wandb_resume to continue training or delete the file to start fresh.")
        return

    # Load preprocessed data
    print("\nLoading preprocessed data...")
    train_data, val_data, _, percentiles = load_preprocessed_data(args)

    if train_data is None:
        print("\n ERROR: Preprocessed data not found!")
        print(f"Please run first: python preprocess_database.py --signal_type {'unipolar' if args.unipolar else 'bipolar'}")
        return

    train_p_inf, train_p_sup = percentiles[0], percentiles[1]

    # ── DF power filter ───────────────────────────────────────────────────────
    if args.min_power > 0.0:
        print(f"\nApplying DF power filter (min_power={args.min_power}) ...")
        train_data = _filter_by_power(train_data, 'train', args)
        val_data   = _filter_by_power(val_data,   'val',   args)

    print(f"\nFinal data shapes:")
    print(f"  Train: {train_data.shape}")
    print(f"  Val: {val_data.shape}")

    # Create data loaders
    train_dataset = TensorDataset(train_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(val_data),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )

    # Create model
    model, criterion, optimizer = compile_model(
        args,
        input_channels=1,
        input_length=train_data.shape[2],
    )
    model.to(device)

    # Print model architecture
    print_model_architecture(model, args)

    # Setup learning rate scheduler and early stopping
    lr_scheduler_ = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.factor_lr,
        patience=args.patience_lr, min_lr=args.min_lr
    )
    early_stopper = EarlyStopping(
        patience=args.patience_es,
        min_delta=args.min_delta,
        restore_best_weights=True
    )

    # Watch model with wandb
    if not args.no_wandb:
        wandb.watch(model, log="all", log_freq=100)

    # Print noise configuration
    if args.add_noise:
        print("\nNoise Augmentation Configuration:")
        if args.random_noise:
            print(f"  Mode: Random selection of {args.min_noise_types}-{args.max_noise_types} noise types per batch")
            print(f"  Available types: gaussian, baseline_wander, powerline, spike")
        else:
            print(f"  Mode: Fixed noise types")
            print(f"  Noise types: {args.noise_types}")
        print(f"  SNR range: [{args.noise_snr_min}, {args.noise_snr_max}] dB")
        print(f"  Sampling frequency: {args.sampling_freq} Hz")
        print(f"  Powerline frequency: {args.powerline_freq} Hz")

    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80 + "\n")

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # Train - handle multi-component loss returns
        train_result = train_epoch(model, train_loader, criterion, optimizer, device, args)
        train_loss, train_r2, train_r2_std = train_result

        # Validate - handle multi-component loss returns
        # Add noise to validation if noise augmentation is enabled
        val_result = validate(
            model, val_loader, criterion, device,
            desc="Validation",
            add_noise=args.add_noise,
            noise_snr_min=args.noise_snr_min,
            noise_snr_max=args.noise_snr_max,
            noise_types=args.noise_types,
            sampling_freq=args.sampling_freq,
            powerline_freq=args.powerline_freq,
            random_noise=args.random_noise,
            min_noise_types=args.min_noise_types,
            max_noise_types=args.max_noise_types,
            kl_beta=args.kl_beta,
        )
        val_loss, val_r2, val_r2_std = val_result

        # Update learning rate
        lr_scheduler_.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Print progress
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f}, R²: {train_r2:.4f}±{train_r2_std:.4f} | "
            f"Val Loss: {val_loss:.4f}, R²: {val_r2:.4f}±{val_r2_std:.4f}"
        )

        # Log to wandb with components
        if not args.no_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/r2": train_r2,
                "train/r2_std": train_r2_std,
                "val/loss": val_loss,
                "val/r2": val_r2,
                "val/r2_std": val_r2_std,
                "learning_rate": current_lr,
            }

            wandb.log(log_dict)

        # Early stopping (check BEFORE updating best_val_loss)
        if early_stopper(model, val_loss, best_val_loss):
            break

        # Save best model (update AFTER early stopping check)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  → Best model saved (Val Loss: {val_loss:.4f})")

        # Generate reconstruction_denoising visualizations every 30 epochs
        if (epoch + 1) % 30 == 0 and not args.no_wandb:
            print(f"\n{'='*80}")
            print(f"GENERATING VISUALIZATIONS - EPOCH {epoch + 1}")
            print(f"{'='*80}")
            visualize_reconstructions(model, train_data, train_p_inf, train_p_sup, f"train_epoch_{epoch+1}", num_samples=5, args=args, device=device)
            visualize_reconstructions(model, val_data, train_p_inf, train_p_sup, f"val_epoch_{epoch+1}", num_samples=5, args=args, device=device)
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)

    visualize_reconstructions(model, train_data, train_p_inf, train_p_sup, f"train", num_samples=5,
                              args=args, device=device)
    visualize_reconstructions(model, val_data, train_p_inf, train_p_sup, f"val", num_samples=5,
                              args=args, device=device)

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()