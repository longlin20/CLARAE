"""
Dominant Frequency (DF) Metric for EGM Signals
==============================================

This module calculates the Dominant Frequency of EGM signals using
Welch's method. Also computes Regularity Index (RI) and Organization Index (OI).

DF = Dominant Frequency in the 3-12 Hz range
RI = Regularity Index (power at DF / total power)
OI = Organization Index (power at DF + harmonics / total power)
"""

import numpy as np
import matplotlib.pyplot as plt

from functions.DFfunctions import calculateDF
from utils import load_patient_signals  # Unified data loading function


# ==============================================================================
# DF Calculation Functions
# ==============================================================================

def calculate_df_batch(signals: np.ndarray, fs: int = 500, verbose: bool = False, show_progress: bool = False,
                       progress_desc: str = "  DF calculation",
                       window_width_seconds: float = None, minDF: float = 3, maxDF: float = 12,
                       Nfft: int = 4096, apply_filters: bool = True, window_overlapping: float = 0.0):
    """
    Calculate Dominant Frequency for multiple signals using Welch's method.

    Args:
        signals: 2D numpy array of shape (num_signals, signal_length)
        fs: Sampling frequency in Hz (default: 500)
        verbose: Print verbose output (default: False)
        window_width_seconds: Window width in seconds for Welch (default: auto based on signal length)
        minDF: Minimum DF in Hz (default: 3)
        maxDF: Maximum DF in Hz (default: 12)
        Nfft: FFT points (default: 4096)
        apply_filters: Apply preprocessing filters (default: True)

    Returns:
        DF_output: DF_class object containing DF values, etc.
    """
    # Auto-calculate window width if not specified
    signal_length = signals.shape[1]
    signal_duration = signal_length / fs

    if window_width_seconds is None:
        window_width_seconds = signal_duration
        #print(f"Auto window_width_seconds: {window_width_seconds:.2f}s (signal: {signal_duration:.2f}s)")

    return calculateDF(signals, fs, verbose=verbose, show_progress=show_progress,
                       progress_desc=progress_desc, window_width_seconds=window_width_seconds,
                       minDF=minDF, maxDF=maxDF, Nfft=Nfft, apply_filters=apply_filters,
                       window_overlapping=window_overlapping)


def calculate_df_error_batch(original: np.ndarray, reconstructed: np.ndarray,
                             fs: int = 500, df_batch: int = 10000,
                             show_progress: bool = True) -> dict:
    """
    Calculate DF absolute error between original and reconstructed signals.

    Processes in mini-batches to avoid excessive memory use.

    Args:
        original: 2D array (N, L) of original signals
        reconstructed: 2D array (N, L) of reconstructed signals
        fs: Sampling frequency in Hz (default: 500)
        df_batch: Signals per mini-batch (default: 10000)
        show_progress: Show tqdm progress bars (default: True)

    Returns:
        dict with 'mean' and 'std' of |DF_orig - DF_recon| across signals
    """
    from tqdm import tqdm

    n_signals = original.shape[0]
    orig_vals, recon_vals = [], []

    for s in tqdm(range(0, n_signals, df_batch), desc="  DF original", unit="batch",
                  disable=not show_progress):
        e = min(s + df_batch, n_signals)
        res = calculate_df_batch(original[s:e], fs=fs, verbose=False, show_progress=False)
        orig_vals.extend(res.DF_values)
        del res

    for s in tqdm(range(0, n_signals, df_batch), desc="  DF reconstructed", unit="batch",
                  disable=not show_progress):
        e = min(s + df_batch, n_signals)
        res = calculate_df_batch(reconstructed[s:e], fs=fs, verbose=False, show_progress=False)
        recon_vals.extend(res.DF_values)
        del res

    errors = np.abs(np.array(orig_vals) - np.array(recon_vals))
    return {'mean': float(np.mean(errors)), 'std': float(np.std(errors))}


def get_df_statistics(df_values: list) -> dict:
    """
    Compute descriptive statistics for DF values.

    Args:
        df_values: List of DF values

    Returns:
        Dictionary with keys: mean, std, min, max, median
    """
    df_array = np.array(df_values)
    return {
        'mean': float(np.mean(df_array)),
        'std': float(np.std(df_array)),
        'min': float(np.min(df_array)),
        'max': float(np.max(df_array)),
        'median': float(np.median(df_array))
    }


def get_ri_statistics(ri_values: list) -> dict:
    """
    Compute descriptive statistics for RI values.

    Args:
        ri_values: List of RI values

    Returns:
        Dictionary with keys: mean, std, min, max, median
    """
    ri_array = np.array(ri_values)
    return {
        'mean': float(np.mean(ri_array)),
        'std': float(np.std(ri_array)),
        'min': float(np.min(ri_array)),
        'max': float(np.max(ri_array)),
        'median': float(np.median(ri_array))
    }


def get_oi_statistics(oi_values: list) -> dict:
    """
    Compute descriptive statistics for OI values.

    Args:
        oi_values: List of OI values

    Returns:
        Dictionary with keys: mean, std, min, max, median
    """
    oi_array = np.array(oi_values)
    return {
        'mean': float(np.mean(oi_array)),
        'std': float(np.std(oi_array)),
        'min': float(np.min(oi_array)),
        'max': float(np.max(oi_array)),
        'median': float(np.median(oi_array))
    }


# ==============================================================================
# Plotting Functions
# ==============================================================================

def plot_signal_with_df(signal: np.ndarray, spectrum: np.ndarray, df_value: float,
                        df_pos: int, fs: int = 500, Nfft: int = 4096,
                        ri: float = None, oi: float = None,
                        title: str = None, save_path: str = None):
    """
    Plot a signal with its power spectrum and mark the Dominant Frequency.

    Args:
        signal: 1D numpy array representing the EGM signal
        spectrum: Normalized power spectrum from calculateDF
        df_value: Dominant Frequency value in Hz
        df_pos: Position of DF in the spectrum
        fs: Sampling frequency in Hz (default: 500)
        Nfft: FFT points used (default: 4096)
        ri: Regularity Index (optional)
        oi: Organization Index (optional)
        title: Optional title for the plot
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # --- Top plot: Time-domain signal ---
    ax1 = axes[0]
    time_ms = np.arange(len(signal)) / fs * 1000
    ax1.plot(time_ms, signal, 'b-', linewidth=0.8)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude (mV)')
    ax1.set_title('EGM Signal (Bipolar)')
    ax1.grid(True, alpha=0.3)

    # --- Bottom plot: Power spectrum with DF marked ---
    ax2 = axes[1]
    freq_axis = np.arange(len(spectrum)) * fs / Nfft
    ax2.plot(freq_axis, spectrum, 'b-', linewidth=1, label='Power Spectrum')

    # Mark DF
    ax2.axvline(x=df_value, color='red', linestyle='--', linewidth=2,
                label=f'DF = {df_value:.2f} Hz')
    ax2.scatter([df_value], [spectrum[df_pos]], color='red', s=100, zorder=5)

    """
    # Add RI and OI text if available
    text_str = f'DF = {df_value:.2f} Hz'
    if ri is not None:
        text_str += f'\nRI = {ri:.3f}'
    if oi is not None:
        text_str += f'\nOI = {oi:.3f}'

    ax2.text(0.95, 0.95, text_str, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    """
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Normalized Power')
    ax2.set_title('Power Spectrum (Welch Method)')
    ax2.set_xlim([0, 25])  # Focus on relevant frequency range
    ax2.legend(loc='upper left')
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
        description='Calculate and plot Dominant Frequency (DF) for EGM signals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python metrics/DF.py --patient_file "DATABASE/EGMS_xz/Patient 2021_09_30_EGMs.xz" --map_index 0
        """
    )
    parser.add_argument('--patient_file', type=str,
                        default='DATABASE/EGMS_xz/Patient 2021_09_30_EGMs.xz',
                        help='Path to .xz patient file')
    parser.add_argument('--map_index', type=int, default=0,
                        help='Index of map to load (default: 0)')
    parser.add_argument('--output_dir', type=str, default='results_plots/DF',
                        help='Output directory for plots (default: results_plots/DF)')
    parser.add_argument('--num_plots', type=int, default=3,
                        help='Number of random signals to plot (default: 3)')
    parser.add_argument('--fs', type=int, default=500,
                        help='Sampling frequency in Hz (default: 500)')

    args = parser.parse_args()

    print("DF Metric - Dominant Frequency Analysis")
    print("=" * 50)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load signals
    print(f"\nLoading signals from: {args.patient_file}")
    signals_uni, patient_id, map_name = load_patient_signals(
        args.patient_file, map_index=args.map_index, signal_type='unipolar'
    )
    signals_bip, _, _ = load_patient_signals(
        args.patient_file, map_index=args.map_index, signal_type='bipolar'
    )
    print(f"  Patient: {patient_id}")
    print(f"  Map: {map_name}")
    print(f"  Unipolar signals shape: {signals_uni.shape}")
    print(f"  Bipolar signals shape: {signals_bip.shape}")

    # Calculate DF for unipolar signals
    print("\nCalculating Dominant Frequency (Unipolar)...")
    df_result_uni = calculate_df_batch(signals_uni, fs=args.fs, verbose=False)

    # Calculate DF for bipolar signals
    print("Calculating Dominant Frequency (Bipolar)...")
    df_result_bip = calculate_df_batch(signals_bip, fs=args.fs, verbose=False)

    # Print statistics for unipolar
    print(f"\n{'='*50}")
    print("UNIPOLAR SIGNALS")
    print(f"{'='*50}")
    print(f"\nDF Statistics (n={df_result_uni.num_DF_signals} signals):")
    print("-" * 40)
    for key, value in get_df_statistics(df_result_uni.DF_values).items():
        print(f"  {key}: {value:.3f} Hz")

    print("\nRI (Regularity Index) Statistics:")
    print("-" * 40)
    for key, value in get_ri_statistics(df_result_uni.DF_RI).items():
        print(f"  {key}: {value:.4f}")

    print("\nOI (Organization Index) Statistics:")
    print("-" * 40)
    for key, value in get_oi_statistics(df_result_uni.DF_OI).items():
        print(f"  {key}: {value:.4f}")

    # Print statistics for bipolar
    print(f"\n{'='*50}")
    print("BIPOLAR SIGNALS")
    print(f"{'='*50}")
    print(f"\nDF Statistics (n={df_result_bip.num_DF_signals} signals):")
    print("-" * 40)
    for key, value in get_df_statistics(df_result_bip.DF_values).items():
        print(f"  {key}: {value:.3f} Hz")

    print("\nRI (Regularity Index) Statistics:")
    print("-" * 40)
    for key, value in get_ri_statistics(df_result_bip.DF_RI).items():
        print(f"  {key}: {value:.4f}")

    print("\nOI (Organization Index) Statistics:")
    print("-" * 40)
    for key, value in get_oi_statistics(df_result_bip.DF_OI).items():
        print(f"  {key}: {value:.4f}")

    # Select random indices for plotting
    num_signals = min(signals_uni.shape[0], signals_bip.shape[0])
    random_indices = np.random.choice(num_signals, size=min(args.num_plots, num_signals), replace=False)

    # Clean map name for filename
    clean_map_name = map_name.replace('/', '_').replace(' ', '_')

    # Plot random unipolar signals
    print(f"\nPlotting {len(random_indices)} random unipolar signals...")
    for idx in random_indices:
        plot_signal_with_df(
            signal=signals_uni[idx],
            spectrum=df_result_uni.DF_Spectrum[idx],
            df_value=df_result_uni.DF_values[idx],
            df_pos=df_result_uni.DF_pos[idx],
            fs=args.fs,
            Nfft=df_result_uni.DF_Nfft[idx],
            ri=df_result_uni.DF_RI[idx],
            oi=df_result_uni.DF_OI[idx],
            title=f'{patient_id} - {map_name} - Unipolar #{idx}',
            save_path=f'{args.output_dir}/{patient_id}_{clean_map_name}_unipolar_{idx}_DF.png'
        )

    # Plot random bipolar signals
    print(f"Plotting {len(random_indices)} random bipolar signals...")
    for idx in random_indices:
        plot_signal_with_df(
            signal=signals_bip[idx],
            spectrum=df_result_bip.DF_Spectrum[idx],
            df_value=df_result_bip.DF_values[idx],
            df_pos=df_result_bip.DF_pos[idx],
            fs=args.fs,
            Nfft=df_result_bip.DF_Nfft[idx],
            ri=df_result_bip.DF_RI[idx],
            oi=df_result_bip.DF_OI[idx],
            title=f'{patient_id} - {map_name} - Bipolar #{idx}',
            save_path=f'{args.output_dir}/{patient_id}_{clean_map_name}_bipolar_{idx}_DF.png'
        )

    print("\nDone!")
