import torch
import numpy as np
import lzma
import pickle

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_patient_signals(xz_path: str, map_index: int = 0, signal_type: str = 'bipolar'):
    """
    Load EGM signals from .xz compressed file.

    This is the unified function for loading patient signals. All metrics modules
    should import this function from utils instead of defining their own.

    Args:
        xz_path: Path to .xz file (e.g., 'DATABASE/EGMS_xz/Patient 2021_09_30_EGMs.xz')
        map_index: Index of the map to load (default: 0)
        signal_type: 'unipolar' or 'bipolar'

    Returns:
        signals: 2D numpy array of shape (num_signals, signal_length)
        patient_id: Patient identifier string
        map_name: Map name string
    """
    with lzma.open(xz_path, 'rb') as f:
        data = pickle.load(f)

    total_maps = len(data)
    if map_index >= total_maps:
        print(f"Warning: map_index {map_index} out of range, using map 0")
        map_index = 0

    row = data.iloc[map_index]
    patient_id = row['patient_id']
    map_name = row['map']

    if signal_type == 'unipolar':
        signal_data = row['unipolar']
    elif signal_type == 'bipolar':
        signal_data = row['bipolar']
    else:
        raise ValueError(f"signal_type must be 'unipolar' or 'bipolar', got '{signal_type}'")

    # Transpose from (signal_length, num_signals) to (num_signals, signal_length)
    signals = np.array(signal_data).T

    return signals, patient_id, map_name


def load_patient_signals_with_info(xz_path: str, map_index: int = 0, signal_type: str = 'unipolar'):
    """
    Load EGM signals from .xz compressed file with additional info.

    Extended version that also returns number of signals and total maps available.

    Args:
        xz_path: Path to .xz file
        map_index: Index of the map to load (default: 0)
        signal_type: 'unipolar' or 'bipolar'

    Returns:
        signals: 2D numpy array of shape (num_signals, signal_length)
        patient_id: Patient identifier string
        map_name: Map name string
        num_signals: Number of signals loaded
        total_maps: Total number of maps available in the file
    """
    with lzma.open(xz_path, 'rb') as f:
        data = pickle.load(f)

    total_maps = len(data)
    if map_index >= total_maps:
        print(f"Warning: map_index {map_index} out of range, using map 0")
        map_index = 0

    row = data.iloc[map_index]
    patient_id = row['patient_id']
    map_name = row['map']

    if signal_type == 'unipolar':
        signal_data = row['unipolar']
    elif signal_type == 'bipolar':
        signal_data = row['bipolar']
    else:
        raise ValueError(f"signal_type must be 'unipolar' or 'bipolar', got '{signal_type}'")

    # Transpose from (signal_length, num_signals) to (num_signals, signal_length)
    signals = np.array(signal_data).T

    return signals, patient_id, map_name, signals.shape[0], total_maps


def list_patient_maps(xz_path: str):
    """
    List all available maps in a patient .xz file.

    Args:
        xz_path: Path to .xz file

    Returns:
        List of dictionaries with map info (index, name, num_points)
    """
    with lzma.open(xz_path, 'rb') as f:
        data = pickle.load(f)

    maps = []
    for idx, row in data.iterrows():
        maps.append({
            'index': idx,
            'name': row['map'],
            'num_points': row['num_points']
        })

    return maps


# =============================================================================
# NOISE FUNCTIONS FOR EGM SIGNAL AUGMENTATION
# =============================================================================
# Reference: Clinical EGM noise sources include:
# 1. Gaussian (thermal/electronic interference)
# 2. Baseline Wander (respiration, electrode drift) - multiple sinusoids
# 3. Powerline (50/60 Hz electrical grid interference)
# 4. Spike Artifacts (electrode movement, contact issues)
#
# Parameters based on literature with SNR-based amplitude control for flexibility.
# =============================================================================

# Available noise types for random selection
NOISE_TYPES = ['gaussian', 'baseline_wander', 'powerline', 'spike']

def add_gaussian_noise(signal, snr_db_min, snr_db_max):
    """
    Add Gaussian white noise to signal with random SNR in dB range.

    Args:
        signal: Input signal tensor [batch, channels, time]
        snr_db_min: Minimum SNR in dB (more noise)
        snr_db_max: Maximum SNR in dB (less noise)

    Returns:
        Noisy signal with same shape as input

    Reference:
        SNR (dB) = 10 * log10(P_signal / P_noise)
        P_noise = P_signal / (10 ^ (SNR_dB / 10))
    """
    # Sample random SNR for this batch (uniform distribution)
    snr_db = torch.FloatTensor(1).uniform_(snr_db_min, snr_db_max).item()

    # Calculate signal power (mean squared value)
    signal_power = torch.mean(signal ** 2)

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10.0)

    # Calculate noise power
    noise_power = signal_power / snr_linear

    # Generate Gaussian noise with calculated power
    noise = torch.randn_like(signal) * torch.sqrt(noise_power)

    # Add noise to signal
    noisy_signal = signal + noise

    return noisy_signal


def add_baseline_wander(signal, snr_db_min, snr_db_max, freq_min=0.01, freq_max=0.3,
                        k_min=1, k_max=4, fs=500):
    """
    Add baseline wander noise to signal using sum of multiple sinusoids.

    Baseline wander is low-frequency noise caused by respiration, electrode movement,
    or patient motion. Modeled as sum of K sinusoids with random frequencies,
    amplitudes, and phases.

    Paper reference parameters:
        - freq_min: 0.01 Hz (minimum sinusoid frequency)
        - freq_max: 0.3 Hz (maximum sinusoid frequency)
        - K: 1-4 sinusoids
        - Amplitude: 0.3-1.0 (relative, scaled by SNR)
        - Phase: 0 to 2*pi

    Args:
        signal: Input signal tensor [batch, channels, time]
        snr_db_min: Minimum SNR in dB (more noise)
        snr_db_max: Maximum SNR in dB (less noise)
        freq_min: Minimum frequency of sinusoids (Hz), default 0.01
        freq_max: Maximum frequency of sinusoids (Hz), default 0.3
        k_min: Minimum number of sinusoids, default 1
        k_max: Maximum number of sinusoids, default 4
        fs: Sampling frequency (Hz), default 1000

    Returns:
        Noisy signal with same shape as input
    """
    batch_size, channels, length = signal.shape
    device = signal.device

    # Sample random SNR
    snr_db = torch.FloatTensor(1).uniform_(snr_db_min, snr_db_max).item()
    signal_power = torch.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    # Generate time vector
    t = torch.linspace(0, length / fs, length, device=device)

    # Generate baseline wander as sum of K low-frequency sinusoids
    noise = torch.zeros_like(signal)
    for b in range(batch_size):
        for c in range(channels):
            # Random number of sinusoids K
            K = np.random.randint(k_min, k_max + 1)
            wander = torch.zeros(length, device=device)

            for _ in range(K):
                # Random frequency within range
                freq = np.random.uniform(freq_min, freq_max)
                # Random amplitude (relative weights, 0.3-1.0 as in paper)
                amp = np.random.uniform(0.3, 1.0)
                # Random phase
                phase = np.random.uniform(0, 2 * np.pi)
                # Add sinusoidal component
                wander = wander + amp * torch.sin(2 * np.pi * freq * t + phase)

            noise[b, c, :] = wander

    # Scale noise to achieve target SNR
    noise = noise * torch.sqrt(noise_power) / (torch.std(noise) + 1e-8)

    return signal + noise


def add_powerline_noise(signal, snr_db_min, snr_db_max, freq=50, fs=500):
    """
    Add powerline interference noise (50/60 Hz) to signal.

    Powerline noise is sinusoidal interference from electrical grid at 50 Hz (Europe)
    or 60 Hz (Americas). Single frequency with random phase.

    Paper reference parameters:
        - freq: 60 Hz (powerline frequency)
        - Amplitude: 0.03-0.1 (relative, scaled by SNR)
        - Phase: 0 to 2*pi

    Args:
        signal: Input signal tensor [batch, channels, time]
        snr_db_min: Minimum SNR in dB (more noise)
        snr_db_max: Maximum SNR in dB (less noise)
        freq: Powerline frequency (50 or 60 Hz), default 50
        fs: Sampling frequency (Hz), default 500

    Returns:
        Noisy signal with same shape as input
    """
    batch_size, channels, length = signal.shape
    device = signal.device

    # Sample random SNR
    snr_db = torch.FloatTensor(1).uniform_(snr_db_min, snr_db_max).item()
    signal_power = torch.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    # Generate time vector
    t = torch.linspace(0, length / fs, length, device=device)

    # Generate powerline noise with random phase
    noise = torch.zeros_like(signal)
    for b in range(batch_size):
        for c in range(channels):
            # Random phase (0 to 2*pi as in paper)
            phase = np.random.uniform(0, 2 * np.pi)
            # Single frequency sinusoid (as in paper)
            powerline = torch.sin(2 * np.pi * freq * t + phase)
            noise[b, c, :] = powerline

    # Scale noise to achieve target SNR
    noise = noise * torch.sqrt(noise_power) / (torch.std(noise) + 1e-8)

    return signal + noise


def add_spike_artifact(signal, snr_db_min, snr_db_max, t_min_ms=2, t_max_ms=1200,
                       amp_min=0.1, amp_max=0.5, num_spikes=1, fs=500):
    """
    Add spike artifacts to signal.

    Spike artifacts are transient, high-amplitude disturbances caused by electrode
    movement, contact issues, or external interference. Modeled as sharp transients
    at random temporal positions.

    Paper reference parameters:
        - t_min: 2 ms (minimum time index for spike)
        - t_max: 40 ms (maximum time index for spike)
        - Amplitude: 0.1-0.8 (relative to signal, scaled by SNR)

    Args:
        signal: Input signal tensor [batch, channels, time]
        snr_db_min: Minimum SNR in dB (more noise)
        snr_db_max: Maximum SNR in dB (less noise)
        t_min_ms: Minimum time position for spike in ms, default 2
        t_max_ms: Maximum time position for spike in ms, default 40
        amp_min: Minimum relative amplitude, default 0.1
        amp_max: Maximum relative amplitude, default 0.8
        num_spikes: Number of spikes to add per signal, default 1
        fs: Sampling frequency (Hz), default 1000

    Returns:
        Noisy signal with same shape as input
    """
    batch_size, channels, length = signal.shape
    device = signal.device

    # Sample random SNR
    snr_db = torch.FloatTensor(1).uniform_(snr_db_min, snr_db_max).item()
    signal_power = torch.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    target_noise_power = signal_power / snr_linear

    # Convert time limits from ms to samples
    t_min_samples = int(t_min_ms * fs / 1000)
    t_max_samples = min(int(t_max_ms * fs / 1000), length - 1)

    noise = torch.zeros_like(signal)

    for b in range(batch_size):
        for c in range(channels):
            for _ in range(num_spikes):
                # Random spike position within time range
                if t_max_samples > t_min_samples:
                    spike_pos = np.random.randint(t_min_samples, t_max_samples)
                else:
                    spike_pos = t_min_samples

                # Random amplitude (positive or negative, within range)
                amplitude = np.random.choice([-1, 1]) * np.random.uniform(amp_min, amp_max)

                # Spike width (wider transient for softer effect)
                width = np.random.randint(5, 10)

                # Create Gaussian-shaped spike
                start = max(0, spike_pos - width)
                end = min(length, spike_pos + width)
                x = torch.arange(start, end, device=device, dtype=torch.float32)
                spike = amplitude * torch.exp(-0.5 * ((x - spike_pos) / (width / 3)) ** 2)
                noise[b, c, start:end] += spike

    # Scale noise to achieve target SNR
    current_power = torch.mean(noise ** 2) + 1e-8
    noise = noise * torch.sqrt(target_noise_power / current_power)

    return signal + noise


def add_random_noise(signal, snr_db_min, snr_db_max, min_types=1, max_types=4,
                     fs=500, powerline_freq=50, return_info=False):
    """
    Add random combination of 1-4 noise types to signal.

    Randomly selects between min_types and max_types noise sources from the
    available pool (gaussian, baseline_wander, powerline, spike) and applies
    them to the signal.

    Args:
        signal: Input signal tensor [batch, channels, time]
        snr_db_min: Minimum SNR in dB (more noise)
        snr_db_max: Maximum SNR in dB (less noise)
        min_types: Minimum number of noise types to apply, default 1
        max_types: Maximum number of noise types to apply, default 4
        fs: Sampling frequency (Hz), default 1000
        powerline_freq: Powerline frequency (50 or 60 Hz), default 60
        return_info: If True, return noise info (types and SNR), default False

    Returns:
        If return_info=False: Noisy signal with same shape as input
        If return_info=True: (noisy_signal, noise_info) where noise_info is dict with
            'types': list of noise type names applied
            'snr_db': the actual SNR used (sampled from range)
    """
    # Randomly select number of noise types to apply
    num_types = np.random.randint(min_types, max_types + 1)

    # Randomly select which noise types to apply
    selected_types = np.random.choice(NOISE_TYPES, size=num_types, replace=False).tolist()

    # Sample a single SNR for this batch
    snr_db = np.random.uniform(snr_db_min, snr_db_max)

    # Adjust SNR for each type: each contributes less individually so combined SNR matches target
    adjusted_snr = snr_db + 10 * np.log10(num_types)

    # Calculate all noises on ORIGINAL signal and accumulate
    total_noise = torch.zeros_like(signal)

    for noise_type in selected_types:
        if noise_type == 'gaussian':
            noise_i = add_gaussian_noise(signal, adjusted_snr, adjusted_snr) - signal
        elif noise_type == 'baseline_wander':
            noise_i = add_baseline_wander(signal, adjusted_snr, adjusted_snr, fs=fs) - signal
        elif noise_type == 'powerline':
            noise_i = add_powerline_noise(signal, adjusted_snr, adjusted_snr,
                                          freq=powerline_freq, fs=fs) - signal
        elif noise_type == 'spike':
            noise_i = add_spike_artifact(signal, adjusted_snr, adjusted_snr, fs=fs) - signal
        total_noise += noise_i

    noisy_signal = signal + total_noise

    if return_info:
        noise_info = {
            'types': selected_types,
            'snr_db': snr_db
        }
        return noisy_signal, noise_info

    return noisy_signal


def add_combined_noise(signal, noise_types, snr_db_min, snr_db_max,
                       fs=500, powerline_freq=50, return_info=False):
    """
    Add specified types of noise to signal.

    Args:
        signal: Input signal tensor [batch, channels, time]
        noise_types: List of noise types to add ['gaussian', 'baseline_wander',
                     'powerline', 'spike']
        snr_db_min: Minimum SNR in dB (more noise)
        snr_db_max: Maximum SNR in dB (less noise)
        fs: Sampling frequency (Hz), default 1000
        powerline_freq: Powerline frequency (50 or 60 Hz), default 60
        return_info: If True, return noise info (types and SNR), default False

    Returns:
        If return_info=False: Noisy signal with same shape as input
        If return_info=True: (noisy_signal, noise_info) where noise_info is dict with
            'types': list of noise type names applied
            'snr_db': the actual SNR used (sampled from range)
    """
    if not noise_types:
        if return_info:
            return signal, {'types': [], 'snr_db': None}
        return signal

    # Sample a single SNR for this batch
    snr_db = np.random.uniform(snr_db_min, snr_db_max)

    # Adjust SNR for each type: each contributes less individually so combined SNR matches target
    n_types = len(noise_types)
    adjusted_snr = snr_db + 10 * np.log10(n_types)

    # Calculate all noises on ORIGINAL signal and accumulate
    total_noise = torch.zeros_like(signal)

    for noise_type in noise_types:
        if noise_type == 'gaussian':
            noise_i = add_gaussian_noise(signal, adjusted_snr, adjusted_snr) - signal
        elif noise_type == 'baseline_wander':
            noise_i = add_baseline_wander(signal, adjusted_snr, adjusted_snr, fs=fs) - signal
        elif noise_type == 'powerline':
            noise_i = add_powerline_noise(signal, adjusted_snr, adjusted_snr,
                                          freq=powerline_freq, fs=fs) - signal
        elif noise_type == 'spike':
            noise_i = add_spike_artifact(signal, adjusted_snr, adjusted_snr, fs=fs) - signal
        total_noise += noise_i

    noisy_signal = signal + total_noise

    if return_info:
        noise_info = {
            'types': list(noise_types),
            'snr_db': snr_db
        }
        return noisy_signal, noise_info

    return noisy_signal


def calculate_lat_matching(lats_orig, lats_recon, tolerance=5):
    """
    Calculate LAT matching between original and reconstructed signals.
    For each original LAT, search for the closest reconstructed LAT within tolerance.
    Each reconstructed LAT can only be assigned to one original LAT (closest wins).
    """
    import numpy as np
    result = {
        'matched_diffs': [],
        'matched_mae_ms': 0.0,
        'n_unmatched_recon': len(lats_recon),
        'n_unmatched_orig': len(lats_orig),
        'n_orig': len(lats_orig),
        'n_recon': len(lats_recon),
    }
    if len(lats_orig) == 0 or len(lats_recon) == 0:
        return result
    used_recon = set()
    for lat_orig in lats_orig:
        best_match = None
        best_dist = tolerance + 1
        for i, lat_recon in enumerate(lats_recon):
            if i in used_recon:
                continue
            dist = abs(lat_orig - lat_recon)
            if dist <= tolerance and dist < best_dist:
                best_match = i
                best_dist = dist
        if best_match is not None:
            result['matched_diffs'].append(best_dist)
            used_recon.add(best_match)
    result['matched_mae_ms'] = (np.mean(result['matched_diffs']) / 500.0 * 1000.0) if result['matched_diffs'] else 0.0
    result['n_unmatched_recon'] = len(lats_recon) - len(used_recon)
    result['n_unmatched_orig'] = len(lats_orig) - len(result['matched_diffs'])
    return result


def r2_score(y_true, y_pred, eps=1e-8):
    """Calculate R2 score."""
    with torch.no_grad():
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        return 1 - ss_res / (ss_tot + eps)

class EarlyStopping:
    """Early stopping callback"""
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.best_model_state = None

    def __call__(self, model, val_loss, best_val_loss):
        if val_loss < best_val_loss - self.min_delta:
            self.wait = 0
            if self.restore_best_weights:
                self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\nEarly stopping triggered after {self.wait} epochs without improvement.")
                if self.restore_best_weights and self.best_model_state:
                    model.load_state_dict(self.best_model_state)
                    print("Best model weights restored.")
                return True
        return False
