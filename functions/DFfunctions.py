import numpy as np
from scipy.fftpack import fft
from tqdm import tqdm
from functions.signalProcessingFunctions import *

#-------------------------------------------------------------------------------------------------
# CLASSES
#-------------------------------------------------------------------------------------------------
# CLASS
class DF_class:

    def __init__(self, num_DF_signals, DF_values, DF_pos, DF_RI, DF_OI, DF_Spectrum, DF_Nfft, DF_fs, DF_All_spectrograms):

        self.num_DF_signals = num_DF_signals
        self.DF_values = DF_values
        self.DF_pos = DF_pos
        self.DF_RI = DF_RI
        self.DF_OI = DF_OI
        self.DF_Spectrum = DF_Spectrum
        self.DF_Nfft = DF_Nfft
        self.DF_fs = DF_fs
        self.DF_All_spectrograms = DF_All_spectrograms

def calculateDF(Bipolar_egms, fs, verbose=False, show_progress=True, progress_desc="  DF calculation", window_width_seconds=4, minDF=3, maxDF=12, Min_freq=3, Max_freq=20, Freq_width=0.75, Nfft=4096, apply_filters=True, window_overlapping=0.0):


    [Nb, L] = Bipolar_egms.shape
    num_DF_signals = Nb

    # Constants
    # window_width_seconds = 4
    # minDF = 3
    # maxDF = 12
    # Min_freq = 3
    # Max_freq = 20
    # Freq_width = 0.75
    # Output signals
    DF_values = []
    DF_pos = []
    DF_RI = []
    DF_OI = []
    DF_Spectrum = []
    DF_Nfft = []
    DF_fs = []
    DF_All_spectrograms = []

    if verbose:
        aux_print = 'Calculating DF...'
        print(aux_print)

    for b in tqdm(range(Nb), desc=progress_desc, disable=not show_progress):
        aux_bipolar_egm = Bipolar_egms[b, :]
        [aux_DF, aux_DF_pos, aux_Norm_Spect, aux_Nfft, aux_all_spectrograms] = calculateDFsingleChannelWelch(aux_bipolar_egm, fs, window_width_seconds,
                                                                                                  minDF=minDF, maxDF=maxDF, verbose=verbose, Nfft=Nfft, apply_filters=apply_filters,
                                                                                                  window_overlapping=window_overlapping)

        [aux_ri, aux_oi] = calculateRIOI(aux_DF, aux_Norm_Spect, fs, aux_Nfft, Min_freq=Min_freq, Max_freq=Max_freq, Freq_width=Freq_width, verbose=verbose)

        # Append DF results
        DF_values.append(aux_DF)
        DF_pos.append(aux_DF_pos)
        DF_Spectrum.append(aux_Norm_Spect)
        DF_Nfft.append(aux_Nfft)
        DF_RI.append(aux_ri)
        DF_OI.append(aux_oi)
        DF_fs.append(fs)
        DF_All_spectrograms.append(aux_all_spectrograms)

    DF_output = DF_class(num_DF_signals, DF_values, DF_pos, DF_RI, DF_OI, DF_Spectrum, DF_Nfft, DF_fs, DF_All_spectrograms)

    return DF_output

def calculateDFsingleChannelWelch(input_signal, fs, window_width_seconds, minDF=3, maxDF=12, verbose=False, Nfft=4096, apply_filters=True, window_overlapping=0.0):
    # fmin        minimum frequency (Hz) for cut analysis
    # fmax        maximum frequency (Hz) for cut analysis
    # DF RANGE
    # minDF       minimum Dominant Frequency (Hz)
    # maxDF       maximum Dominant Frequency (Hz)
    # window_overlapping  fraction of overlap: 0.0 = no overlap, 0.5 = 50% overlap

    # OUTPUTS: [DF, DF_pos, Norm_Spect, F_ref]

    if verbose:
        aux_print = '- Calculating DF (Welch)'
        print(aux_print)

    L = len(input_signal)
    #Nfft = np.power(2, 12)  # FFT points - 4096

    # Apply filters!
    # ABS
    aux_signal = input_signal
    if apply_filters:
        aux_signal = np.abs(input_signal)
        # LPF
        aux_signal = egm_filter(aux_signal, fs=fs, fc=20, f_order=3, btype='lowpass', med_filter=False, select_axis=0)

    # Num. Windows
    window_width_samples = int(window_width_seconds * fs)
    window_overlapping_samples = int(window_overlapping * window_width_samples)
    num_windows = int(np.ceil(L / (window_width_samples - window_overlapping_samples)))

    if verbose:
        print('  - window_width_samples: ', window_width_samples)
        print('  - window_overlapping: ', window_overlapping)
        print('  - window_overlapping_samples: ', window_overlapping_samples)
        print('  - num_windows: ', num_windows)

    # PRE-PROCESS EGMs:
    DF_window = []
    Norm_Spect_window = []
    DF_pos_w = []

    w_valid_index = 0
    aux_spectrogram = 0

    All_spectrograms = np.zeros((num_windows, int(Nfft / 2)))

    for w in range(num_windows):

        if w == 0:
            window_init_index = 0
            window_end_index = window_init_index + window_width_samples - 1
            window_last_window_end_index = window_end_index
        else:
            window_init_index = window_last_window_end_index - window_overlapping_samples + 1
            window_end_index = window_init_index + window_width_samples - 1
            window_last_window_end_index = window_end_index

        if verbose:
            print('   - w: ', w)
            print('   - window_init_index: ', window_init_index)
            print('   - window_end_index: ', window_end_index)

        window_indices = np.linspace(window_init_index, window_end_index, window_end_index - window_init_index + 1)
        window_indices = window_indices.astype(int)

        if window_indices[-1] <= L:
            w_valid_index = w
            aux_signal_segment = aux_signal[window_indices[0]:window_indices[-1] + 1]

            # Apply HANNING WINDOW
            aux_hann = np.hanning(len(window_indices))
            aux_hann = aux_hann * aux_signal_segment

            # Find DF
            aux_FFT = np.abs(fft(aux_hann, n=Nfft))  # |fft(x)|
            aux_FFT = aux_FFT[range(int(Nfft / 2))]  # only positive freqs

            All_spectrograms[w, :] = aux_FFT

            # Accumulate the windowed spectrograms
            aux_spectrogram = aux_spectrogram + aux_FFT

            # ONCE ALL THE WINDOWS ARE PROCESS TAKE THE MEAN SPECTROGRAM
    aux_spectrogram = aux_spectrogram / (w_valid_index + 1)

    # Normalize the spectrogram
    aux_spectrogram = aux_spectrogram / np.sum(aux_spectrogram)

    aux_min_df_index = int(round(minDF * Nfft / fs))
    aux_max_df_index = int(round(maxDF * Nfft / fs))

    aux_FFT_DF_range = aux_spectrogram[aux_min_df_index:aux_max_df_index]
    aux_pos_DF = aux_FFT_DF_range.argmax()

    DF_pos = aux_pos_DF + aux_min_df_index
    DF = DF_pos * fs / Nfft
    Norm_Spect = aux_spectrogram

    if verbose:
        print('   - DF: ', DF)
        print('   - aux_min_df_index: ', aux_min_df_index)
        print('   - aux_max_df_index: ', aux_max_df_index)
        print('   - aux_spectrogram: ', aux_spectrogram)
        print('   - aux_spectrogram.shape: ', aux_spectrogram.shape)

    return DF, DF_pos, Norm_Spect, Nfft, All_spectrograms


def calculateRIOI(DF, Spectrum, fs, Nfft, Min_freq=3, Max_freq=20, Freq_width=0.75, verbose=False):

    if verbose:
        aux_print = '- Calculating RI/OI'
        print(aux_print)

    # Min_freq = 3  # Hz
    # Max_freq = 20  # Hz
    Min_freq_pos = int(Min_freq * Nfft / fs)
    Max_freq_pos = int(Max_freq * Nfft / fs)
    # Freq_width = 0.75  # Hz
    freq_width_pos = int(Freq_width * Nfft / fs)

    use_half_harmonics = 0

    if DF > 5:
        use_half_harmonics = 1

    aux_OI_power = 0
    aux_RI_power = 0

    for i in range(3):

        if use_half_harmonics:
            # 0.5·DF, DF, 1.5·DF, 2·DF
            aux_f = (1 + i) * DF / 2
        else:
            # DF, 2·DF, 3·DF, 4·DF
            aux_f = (1 + i) * DF

        if aux_f >= Min_freq and aux_f <= Max_freq:
            # Is in the 3-20Hz band
            aux_harmonic_pos = int(aux_f * Nfft / fs)
            aux_harmonic_spect = Spectrum[aux_harmonic_pos - freq_width_pos:aux_harmonic_pos + freq_width_pos + 1]
            aux_harmonic_power = np.sum(aux_harmonic_spect)

            if verbose:
                print('  - Harmonic Freq: ', aux_f)
                print('  - Harmonic Pos: ', aux_harmonic_pos)
                print('  - Harmonic power: ', aux_harmonic_power)
                print('  - Harmonic Spect: ', aux_harmonic_spect)

            # Store DF + Harmonics power for OI
            aux_OI_power = aux_OI_power + aux_harmonic_power

            if i == 0:
                # Store only DF power for RI
                aux_RI_power = aux_harmonic_power

    Total_power = np.sum(Spectrum[Min_freq_pos:Max_freq_pos + 1])
    if verbose:
        print('  - Total Power: ', Total_power)

    RI = aux_RI_power / Total_power
    OI = aux_OI_power / Total_power

    if verbose:
        print('  - RI = ', RI)
        print('  - OI = ', OI)

    #print('TO BE DONE - Check if RI OI are calculated correctly!')

    return RI, OI