"""
Peak-to-Peak Voltage (Vpp) Metric for EGM Signals
==================================================

This module provides functions to calculate the peak-to-peak voltage (Vpp)
of EGM signals. Vpp is defined as the difference between the maximum and
minimum values of a signal.

Vpp = max(signal) - min(signal)

Example: A signal ranging from -1.5 to +2.0 mV has Vpp = 3.5 mV
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import load_patient_signals  # Unified data loading function


def calculate_vpp_batch(signals: np.ndarray) -> np.ndarray:
    """
    Calculate peak-to-peak voltage for multiple signals.

    Args:
        signals: 2D numpy array of shape (num_signals, signal_length)

    Returns:
        1D numpy array of Vpp values, one per signal
    """
    return np.max(signals, axis=1) - np.min(signals, axis=1)


def calculate_vpp_error_batch(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """
    Calculate Vpp absolute error between original and reconstructed signals.

    Args:
        original: 2D array (N, L) of original signals
        reconstructed: 2D array (N, L) of reconstructed signals

    Returns:
        dict with 'mean' and 'std' of |Vpp_orig - Vpp_recon| across signals
    """
    vpp_orig = calculate_vpp_batch(original)
    vpp_recon = calculate_vpp_batch(reconstructed)
    errors = np.abs(vpp_orig - vpp_recon)
    return {'mean': float(np.mean(errors)), 'std': float(np.std(errors))}


def get_vpp_statistics(vpp_values: np.ndarray) -> dict:
    """
    Compute descriptive statistics for Vpp values.

    Args:
        vpp_values: 1D numpy array of Vpp values

    Returns:
        Dictionary with keys: mean, std, min, max, median
    """
    return {
        'mean': float(np.mean(vpp_values)),
        'std': float(np.std(vpp_values)),
        'min': float(np.min(vpp_values)),
        'max': float(np.max(vpp_values)),
        'median': float(np.median(vpp_values))
    }


def plot_signal_with_vpp(signal: np.ndarray, fs: int = 500, title: str = None,
                         save_path: str = None):
    """
    Plot a single signal with its Vpp marked (max and min points).

    Args:
        signal: 1D numpy array representing the EGM signal
        fs: Sampling frequency in Hz (default: 500)
        title: Optional title for the plot
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # Time axis in ms
    time_ms = np.arange(len(signal)) / fs * 1000

    # Plot signal
    ax.plot(time_ms, signal, 'b-', linewidth=1, label='Signal')

    # Find max and min
    max_idx = np.argmax(signal)
    min_idx = np.argmin(signal)
    max_val = signal[max_idx]
    min_val = signal[min_idx]
    vpp = max_val - min_val

    # Mark max and min points
    ax.scatter(time_ms[max_idx], max_val, color='red', s=100, zorder=5,
               label=f'Max: {max_val:.2f} mV')
    ax.scatter(time_ms[min_idx], min_val, color='green', s=100, zorder=5,
               label=f'Min: {min_val:.2f} mV')

    # Draw vertical arrow showing Vpp
    mid_time = (time_ms[max_idx] + time_ms[min_idx]) / 2
    ax.annotate('', xy=(mid_time, max_val), xytext=(mid_time, min_val),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax.text(mid_time + 20, (max_val + min_val) / 2, f'Vpp = {vpp:.2f} mV',
            fontsize=12, color='purple', fontweight='bold')

    # Horizontal dashed lines at max/min
    ax.axhline(y=max_val, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=min_val, color='green', linestyle='--', alpha=0.5)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_title(title or f'EGM Signal - Vpp = {vpp:.2f} mV')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to: {save_path}")


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser(
        description='Calculate and plot Vpp for EGM signals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python metrics/Vpp.py --patient_file "DATABASE/EGMS_xz/Patient 2021_09_30_EGMs.xz" --map_index 2 --output_dir results_plots/Vpp
        """
    )
    parser.add_argument('--patient_file', type=str,
                        default='DATABASE/EGMS_xz/Patient 2021_09_30_EGMs.xz',
                        help='Path to .xz patient file')
    parser.add_argument('--map_index', type=int, default=0,
                        help='Index of map to load (default: 0)')
    parser.add_argument('--output_dir', type=str, default='results_plots/Vpp',
                        help='Output directory for plots (default: results_plots/Vpp)')
    parser.add_argument('--num_plots', type=int, default=3,
                        help='Number of random signals to plot per type (default: 3)')
    parser.add_argument('--fs', type=int, default=500,
                        help='Sampling frequency in Hz (default: 500)')

    args = parser.parse_args()

    print("Vpp Metric - Example with Real Patient Data")
    print("=" * 50)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load unipolar signals
    print(f"\nLoading signals from: {args.patient_file}")
    signals_uni, patient_id, map_name = load_patient_signals(
        args.patient_file, map_index=args.map_index, signal_type='unipolar'
    )
    print(f"  Patient: {patient_id}")
    print(f"  Map: {map_name}")
    print(f"  Unipolar signals shape: {signals_uni.shape}")

    # Load bipolar signals
    signals_bip, _, _ = load_patient_signals(
        args.patient_file, map_index=args.map_index, signal_type='bipolar'
    )
    print(f"  Bipolar signals shape: {signals_bip.shape}")

    # Calculate Vpp statistics
    vpp_uni = calculate_vpp_batch(signals_uni)
    vpp_bip = calculate_vpp_batch(signals_bip)

    print("\nUnipolar Vpp Statistics:")
    for key, value in get_vpp_statistics(vpp_uni).items():
        print(f"  {key}: {value:.3f} mV")

    print("\nBipolar Vpp Statistics:")
    for key, value in get_vpp_statistics(vpp_bip).items():
        print(f"  {key}: {value:.3f} mV")

    # Select random indices
    num_signals = min(signals_uni.shape[0], signals_bip.shape[0])
    random_indices = np.random.choice(num_signals, size=min(args.num_plots, num_signals), replace=False)

    # Clean map name for filename
    clean_map_name = map_name.replace('/', '_').replace(' ', '_')

    # Plot random unipolar signals
    print(f"\nPlotting {len(random_indices)} random unipolar signals...")
    for idx in random_indices:
        plot_signal_with_vpp(
            signals_uni[idx],
            fs=args.fs,
            title=f'{patient_id} - {map_name} - Unipolar #{idx}',
            save_path=f'{args.output_dir}/{patient_id}_{clean_map_name}_unipolar_{idx}.png'
        )

    # Plot random bipolar signals
    print(f"Plotting {len(random_indices)} random bipolar signals...")
    for idx in random_indices:
        plot_signal_with_vpp(
            signals_bip[idx],
            fs=args.fs,
            title=f'{patient_id} - {map_name} - Bipolar #{idx}',
            save_path=f'{args.output_dir}/{patient_id}_{clean_map_name}_bipolar_{idx}.png'
        )

    print("\nDone!")
