"""
Unipolar EGM LAT Detection Script
==================================

This script loads unipolar EGM signals from compressed .xz files and detects
Local Activation Times (LATs) using slope-based detection with exponential
decay thresholds.

Generates plots similar to reference paper:
  - Original signals: Unprocessed EGM signals
  - Slope approximation β[n]
  - LAT detection with threshold and detected peaks
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from joblib import Parallel, delayed
import multiprocessing

import numpy as np

from functions.signalProcessingFunctions import (
    LATSettingsClass,
    calculateUnipolarEGMSlopeParallel,
    detectLATs
)
from utils import load_patient_signals_with_info, list_patient_maps, calculate_lat_matching  # Unified data loading


def load_patient_unipolar_signals(xz_path, map_index=0):
    """
    Load unipolar EGM signals from .xz compressed file.

    This is a wrapper around load_patient_signals_with_info that also
    prints map information for backwards compatibility.

    Args:
        xz_path: Path to .xz file (e.g., 'DATABASE/EGMS_xz/Patient 2019_09_30_EGMs.xz')
        map_index: Index of the map to load (default: 0, first map in file)

    Returns:
        signals: 2D numpy array of shape (num_signals, signal_length)
        patient_id: Patient identifier string
        map_name: Map name string
        num_signals: Number of signals loaded
        total_maps: Total number of maps available in the file
    """
    # List available maps
    maps = list_patient_maps(xz_path)
    print(f"  Total maps available in file: {len(maps)}")
    for m in maps:
        print(f"    Map {m['index']}: {m['name']} ({m['num_points']} signals)")

    # Load signals using unified function
    return load_patient_signals_with_info(xz_path, map_index=map_index, signal_type='unipolar')


# ==============================================================================
# Signal Processing Functions
# ==============================================================================

def calculate_slopes_batch(signals, m, fs=1000):
    """
    Calculate slope approximation for multiple signals in parallel

    Args:
        signals: 2D array of shape (num_signals, signal_length)
        m: Slope filter order (samples)
        fs: Sampling frequency (Hz) - not used directly but kept for consistency

    Returns:
        slopes: 2D array of shape (num_signals, signal_length) with slope values
    """
    # Use parallel implementation from signalProcessingFunctions
    slopes = calculateUnipolarEGMSlopeParallel(signals, m)
    return slopes


def detect_lats_batch(signals, slopes, m, fs):
    """
    Detect LATs for multiple signals in parallel

    Args:
        signals: 2D array of shape (num_signals, signal_length)
        slopes: 2D array of shape (num_signals, signal_length)
        m: Slope filter order
        fs: Sampling frequency (Hz)

    Returns:
        lat_detections: List of LATDetectionClass objects, one per signal
    """
    # Create LATSettings object (same for all signals)
    lat_settings = LATSettingsClass(
        M=m,
        fs=fs,
        tau_input=0.00035,  # default exponential decay
        blank_period_time=0.045,  # 90ms minimum between LATs
        sigma_abs_th=0.05  # minimum activity threshold
    )

    # Calculate beta slopes β+[n] for all signals
    beta_slopes = -1 * slopes
    beta_slopes[beta_slopes < 0] = 0

    # Detect LATs in parallel (similar to robustUnipolarEGMLATdetection)
    num_cores = multiprocessing.cpu_count()
    lat_detections = Parallel(n_jobs=num_cores, verbose=0)(
        delayed(detectLATs)(beta_slopes[i, :], lat_settings)
        for i in range(signals.shape[0])
    )

    return lat_detections


def calculate_lat_metrics_batch(original: np.ndarray, reconstructed: np.ndarray,
                                fs: int = 500, m: int = 10,
                                lat_batch_size: int = 500,
                                tolerance: int = 5) -> dict:
    """
    Calculate LAT detection metrics comparing original vs reconstructed signals.

    Processes in mini-batches to limit memory use.

    Args:
        original: 2D array (N, L) of original signals
        reconstructed: 2D array (N, L) of reconstructed signals
        fs: Sampling frequency in Hz (default: 500)
        m: Slope filter order in samples (default: 5)
        lat_batch_size: Signals per mini-batch (default: 500)
        tolerance: Matching tolerance in samples (default: 5)

    Returns:
        dict with keys:
            'LAT_matched_MAE_ms'  : {'mean', 'std'}
            'LAT_unmatched_recon' : {'mean', 'std'}
            'LAT_unmatched_orig'  : {'mean', 'std'}
    """
    from tqdm import tqdm

    lat_settings = LATSettingsClass(
        M=m, fs=fs,
        tau_input=0.00035,
        blank_period_time=0.045,
        sigma_abs_th=0.05,
    )

    n_signals = original.shape[0]
    all_mae_ms = []
    all_unmatched_recon = []
    all_unmatched_orig = []

    pbar = tqdm(total=n_signals, desc="  LAT detection")
    for start in range(0, n_signals, lat_batch_size):
        end = min(start + lat_batch_size, n_signals)

        slopes_orig = calculateUnipolarEGMSlopeParallel(original[start:end], m)
        slopes_recon = calculateUnipolarEGMSlopeParallel(reconstructed[start:end], m)

        beta_orig = -1 * slopes_orig
        beta_orig[beta_orig < 0] = 0
        del slopes_orig
        beta_recon = -1 * slopes_recon
        beta_recon[beta_recon < 0] = 0
        del slopes_recon

        for i in range(end - start):
            lat_det_orig = detectLATs(beta_orig[i], lat_settings)
            lat_det_recon = detectLATs(beta_recon[i], lat_settings)
            lat_match = calculate_lat_matching(
                lat_det_orig.activation_peaks_indices.astype(int),
                lat_det_recon.activation_peaks_indices.astype(int),
                tolerance=tolerance
            )
            all_mae_ms.append(lat_match['matched_mae_ms'])
            all_unmatched_recon.append(lat_match['n_unmatched_recon'])
            all_unmatched_orig.append(lat_match['n_unmatched_orig'])
            pbar.update(1)

        del beta_orig, beta_recon
    pbar.close()

    mae = np.array(all_mae_ms)
    unm_r = np.array(all_unmatched_recon)
    unm_o = np.array(all_unmatched_orig)

    return {
        'LAT_matched_MAE_ms':  {'mean': float(np.mean(mae)),   'std': float(np.std(mae)),   'values': mae},
        'LAT_unmatched_recon': {'mean': float(np.mean(unm_r)), 'std': float(np.std(unm_r)), 'values': unm_r},
        'LAT_unmatched_orig':  {'mean': float(np.mean(unm_o)), 'std': float(np.std(unm_o)), 'values': unm_o},
    }


# ==============================================================================
# Plotting Functions
# ==============================================================================


def plot_single_signal_complete(time_ms, signal, slope, positive_slope, lat_detection, signal_idx):
    """
    Plot complete analysis for a single signal: Original, Slope, and LATs in 3 rows

    Args:
        time_ms: Time array in milliseconds
        signal: 1D array (signal_length) - original signal
        slope: 1D array (signal_length) - slope approximation
        positive_slope: 1D array (signal_length) - positive slope only
        lat_detection: LATDetectionClass object
        signal_idx: Signal index (for title)

    Returns:
        fig: matplotlib figure object
    """
    # Create figure with 3 rows (subplots stacked vertically)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Row 1: Original Signal with LAT markers
    ax_orig = axes[0]
    ax_orig.plot(time_ms, signal, linewidth=1, color='black')

    # Mark LATs on original signal
    if len(lat_detection.activation_peaks_indices) > 0:
        lat_indices = np.array(lat_detection.activation_peaks_indices, dtype=int)
        lat_times = time_ms[lat_indices]
        lat_values_orig = signal[lat_indices]

        # Vertical lines for LATs
        for lat_time in lat_times:
            ax_orig.axvline(x=lat_time, color='red', linewidth=1.5, alpha=0.6)

        # Add markers
        ax_orig.scatter(lat_times, lat_values_orig, color='red', s=60,
                      marker='*', zorder=5, label='LATs')
        ax_orig.legend(loc='upper right')

    ax_orig.set_ylabel('Amplitude (mV)')
    ax_orig.set_title('Original Signal')
    ax_orig.grid(True, alpha=0.3)

    # Row 2: Slope Approximation
    ax_slope = axes[1]
    ax_slope.plot(time_ms, slope, linewidth=1, color='blue')
    ax_slope.set_ylabel('Slope β[n] (mV/s)')
    ax_slope.set_title('Slope Approximation')
    ax_slope.grid(True, alpha=0.3)
    ax_slope.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    # Row 3: LAT Detection
    ax_lat = axes[2]

    # Plot positive slopes β+[n]
    ax_lat.plot(time_ms, positive_slope, linewidth=1, color='blue', label='β+[n]')

    # Plot threshold
    ax_lat.plot(time_ms, lat_detection.threshold, linewidth=1, color='green',
               linestyle='--', label='Threshold')

    # Mark detected LATs
    if len(lat_detection.activation_peaks_indices) > 0:
        lat_indices = np.array(lat_detection.activation_peaks_indices, dtype=int)
        lat_times = time_ms[lat_indices]
        lat_values = positive_slope[lat_indices]

        # Vertical lines
        y_min = min(0, positive_slope.min())
        signal_max = positive_slope.max()
        lat_values_clipped = np.minimum(lat_values, signal_max)
        ax_lat.vlines(lat_times, ymin=y_min, ymax=lat_values_clipped, colors='red',
                    linewidth=1.5, alpha=0.6)

        # Add ellipsis where lines are clipped
        for t, v_orig in zip(lat_times, lat_values):
            if v_orig > signal_max:
                ax_lat.annotate('...', xy=(t, signal_max), fontsize=10, ha='center', va='bottom')

        # Markers (also clipped to signal_max)
        lat_values_marker = np.minimum(lat_values, signal_max)
        ax_lat.scatter(lat_times, lat_values_marker, color='red', s=60, marker='*',
                     zorder=5, label='LATs')

    ax_lat.set_xlabel('Time (ms)')
    ax_lat.set_ylabel('Amplitude (mV)')
    ax_lat.set_title(f'LAT Detection ({len(lat_detection.activation_peaks_indices)} LATs)')
    ax_lat.grid(True, alpha=0.3)
    ax_lat.legend(loc='upper right')

    # Limit Y axis to signal maximum and ensure max value appears in ticks
    y_min_axis = min(0, positive_slope.min())
    signal_max_axis = positive_slope.max()
    ax_lat.set_ylim(bottom=y_min_axis, top=signal_max_axis * 1.05)

    # Ensure the max value appears in Y axis ticks
    current_yticks = list(ax_lat.get_yticks())
    if signal_max_axis not in current_yticks:
        current_yticks.append(signal_max_axis)
    ax_lat.set_yticks([t for t in current_yticks if y_min_axis <= t <= signal_max_axis])

    fig.suptitle(f'Signal {signal_idx + 1} - Analysis', fontsize=14)
    plt.tight_layout()

    return fig


# ==============================================================================
# Main Execution
# ==============================================================================

def main(patient_file=None, map_index=0, num_signals=None, num_plot=5, m=10, fs=500,
         output_dir='results_plots/LATs'):
    """
    Main execution function

    Args:
        patient_file: Path to .xz file (default: DATABASE/EGMS_xz/Patient 2019_09_30_EGMs.xz)
        map_index: Index of map to load from file (default: 0)
        num_signals: Number of signals to process (default: None = all)
        num_plot: Number of signals to plot (default: 5)
        m: Slope filter order in samples (default: 10)
        fs: Sampling frequency in Hz (default: 500)
        output_dir: Output directory for plots (default: results_plots/LATs)
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load signals from .xz file
    print(f"Loading unipolar signals from: {patient_file}")
    signals, patient_id, map_name, total_signals, total_maps = \
        load_patient_unipolar_signals(patient_file, map_index=map_index)

    print(f"\nSelected:")
    print(f"  Patient: {patient_id}")
    print(f"  Map: {map_name} (index {map_index})")
    print(f"  Total signals in map: {total_signals}")

    # Select subset of signals if specified
    if num_signals is not None and num_signals < total_signals:
        signals = signals[:num_signals]
        print(f"  Processing first {num_signals} signals")
    else:
        num_signals = total_signals
        print(f"  Processing all {num_signals} signals")

    signal_length = signals.shape[1]
    time_ms = np.arange(signal_length) / fs * 1000  # Convert to ms

    print(f"\nSignal info:")
    print(f"  Length: {signal_length} samples")
    print(f"  Duration: {signal_length/fs:.2f} s")
    print(f"  Min: {signals.min():.3f} mV, Max: {signals.max():.3f} mV")
    print(f"  Mean: {signals.mean():.3f} mV, Std: {signals.std():.3f} mV")

    # Calculate slopes for all signals (parallel processing)
    m_ms = m / fs * 1000  # Convert to milliseconds
    print(f"\nCalculating slope approximations (M={m} samples = {m_ms:.1f} ms)...")
    slopes = calculate_slopes_batch(signals, m=m, fs=fs)
    print(f"  Completed: {slopes.shape[0]} signals processed")

    # Detect LATs for all signals (parallel processing)
    print("\nDetecting LATs...")
    lat_detections = detect_lats_batch(signals, slopes, m=m, fs=fs)

    # Print statistics
    num_lats_per_signal = [len(det.activation_peaks_indices) for det in lat_detections]
    total_lats = sum(num_lats_per_signal)
    print(f"  Total LATs detected: {total_lats}")
    print(f"  LATs per signal: mean={np.mean(num_lats_per_signal):.1f}, "
          f"min={np.min(num_lats_per_signal)}, max={np.max(num_lats_per_signal)}")

    # Calculate positive slopes for plotting
    beta_slopes = -1 * slopes
    beta_slopes[beta_slopes < 0] = 0

    # Create plots (plot subset of signals)
    num_plot = min(num_plot, num_signals)

    # Clean map name for filename (replace special characters)
    clean_map_name = map_name.replace('/', '_').replace(' ', '_')

    print(f"\nCreating plots for {num_plot} signals...")

    # Create individual plot for each signal
    plot_filenames = []
    for i in range(num_plot):
        #print(f"  - Plotting signal {i+1}/{num_plot}...")

        # Create plot for single signal
        fig = plot_single_signal_complete(
            time_ms,
            signals[i],
            slopes[i],
            beta_slopes[i],
            lat_detections[i],
            signal_idx=i
        )

        # Save plot
        plot_filename = f'{output_dir}/{patient_id}_{clean_map_name}_m{m}_s{i+1:02d}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plot_filenames.append(plot_filename)

    print(f"\nPlots saved to {output_dir}/")
    for filename in plot_filenames:
        print(f"  {filename}")

    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detect LATs in unipolar EGM signals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  # Full example
 python LATs_unipolar.py --patient_file "DATABASE/EGMS_xz/Patient 2021_09_30_EGMs.xz" --map_index 2 --num_signals 10 --num_plot 5 --m 10
        """
    )

    parser.add_argument('--patient_file', type=str, default='DATABASE/EGMS_xz/Patient 2021_09_30_EGMs.xz',
                       help='Path to .xz patient file (default: DATABASE/EGMS_xz/Patient 2021_09_30_EGMs.xz)')
    parser.add_argument('--map_index', type=int, default=0,
                       help='Index of map to load from file (default: 0)')
    parser.add_argument('--output_dir', type=str, default='results_plots/LATs',
                       help='Output directory for plots (default: results_plots/LATs)')
    parser.add_argument('--num_signals', type=int, default=5,
                       help='Number of signals to process (default: None = all signals in map)')
    parser.add_argument('--num_plot', type=int, default=3,
                       help='Number of signals to plot (default: 3)')
    parser.add_argument('--m', type=int, default=5,
                       help='Slope filter order in samples (default: 5)')
    parser.add_argument('--fs', type=int, default=500,
                       help='Sampling frequency in Hz (default: 500)')

    args = parser.parse_args()

    main(patient_file=args.patient_file,
         map_index=args.map_index,
         output_dir=args.output_dir,
         num_signals=args.num_signals,
         num_plot=args.num_plot,
         m=args.m,
         fs=args.fs)
