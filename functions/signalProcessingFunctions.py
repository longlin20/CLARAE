"""Help on built-in module signalProcessingFunctions:

NAME
    signalProcessingFunctions

DESCRIPTION
    This module provides functions to process AF signals

CLASSES
    LATSettingsClass
    LATDetectionClass

FUNCTIONS
    calculateUnipolarEGMSlope(egm_signal, m)
        Calculates the m order slope approximation of the EGM signal(s)
        Returns the slope approximation (filtered signal)

    calculateUnipolarEGMSlopeParallel(egm_signal, m)
        Calculates the m order slope approximation of the EGM signal(s)
        For loop is executed in parallel
        Returns the slope approximation (filtered signal)

    plotUnipolarLATDetection(unipolar_signals, slope_signals, LATDetection)
        Plots LAT detection signals
        Returns None

    robustUnipolarEGMLATdetection(Bipolar_egms, Unipolar_egms, Bipolar_relative_indices, Unipolar_relative_indices, LATSettings_bipolar, LATSettings_unipolar, show_figures, verbose)
        Calculates robust LAT detection for one bipolar EGM using unipolar and bipolar deflections
        Returns robust bipolar LATs, the LAT interval durations of the EGM and the dominant (mode) cycle of the bipolar EGM
"""

# signalProcessingFunctions
import math
import numpy as np
import scipy.signal as ss
# from scipy.signal import filtfilt, butter, lfilter
from scipy.signal import medfilt
from scipy.signal import savgol_filter
import configparser # read kivy .ini file
from scipy import stats

from joblib import Parallel, delayed
import multiprocessing

#-------------------------------------------------------------------------------------------------
# CLASSES
#-------------------------------------------------------------------------------------------------
# CLASS LATSettingsClass
class LATSettingsClass:
    """
    A class used to store Local Activation Time (LAT) detection settings

    Attributes
    ----------
    M : int
        Slope filter order
    fs: int
        Sampling frequency
    tau_input : float
        Exponential decay constant
    blank_period_time : float
        Minimum blank period between consecutive local activations in seconds
    T_prev : float
        Time offset to apply LAT detection in seconds
    sigma_abs_th : float
        Minimum activity detection threshold
    LATs_search : int
        Maximum search window for robust LAT detection

    Methods
    -------
    ...(...)
        Calculates...
    """

    def __init__(self, M=10, fs=1000, tau_input=0.00035, blank_period_time=0.090, T_prev=0, sigma_abs_th=0.02, LATs_search=20):

        self.M = M
        self.fs = fs
        self.tau_input = tau_input
        self.blank_period_time = blank_period_time
        self.T_prev = T_prev
        self.sigma_abs_th = sigma_abs_th
        self.LATs_search = LATs_search

# CLASS LATDetectionClass
class LATDetectionClass:
    """
    A class used to store Local Activation Time (LAT) detection results

    Attributes
    ----------
    LATSettings : LATSettingsClass
        LATSettingsClass object containing the LAT settings employed
    activation_peaks_values : (numpy float array)
        EGM value at the LATs
    activation_peaks_indices : (numpy float array)
        Time instant for the LATs
    threshold : (numpy float array)
        Theshold
    abs_threshold : (numpy float array)
        Threshold
    isocro : (numpy float array)
        Elapsed time since last LAT detected

    Methods
    -------
    ...(...)
        Calculates...
    """
    def __init__(self, LATSettings, activation_peaks_values, activation_peaks_indices, threshold, abs_threshold, isocro):

        self.LATSettings = LATSettings
        self.activation_peaks_values = activation_peaks_values
        self.activation_peaks_indices = activation_peaks_indices
        self.threshold = threshold
        self.abs_threshold = abs_threshold
        self.isocro = isocro

#-------------------------------------------------------------------------------------------------
# FUNCTIONS
#-------------------------------------------------------------------------------------------------
# FUNCTION egm_filter
def egm_filter(X, fs, fc, f_order=3, btype='lowpass', med_filter=False, select_axis=1):

    # Create an n-order lowpass butterworth filter.
    aux_nyq = 0.5*fs
    try:
        if len(fc) == 2:
            aux_fc = [fc[0] / aux_nyq, fc[1] / aux_nyq]

            aux_print = 'nyq = ' + str(aux_nyq) + ', fc=' + str(aux_fc[0]) + ', ' + str(aux_fc[1])
    except:
        aux_fc = fc / aux_nyq

        aux_print = 'nyq = ' + str(aux_nyq) + ', fc=' + str(aux_fc)

    b, a = ss.butter(f_order, aux_fc, btype=btype)
    # Use filtfilt to apply the filter.
    Y = ss.filtfilt(b, a, X, axis=select_axis)
    # Y = ss.lfilter(b,a,X,axis=1)
    # Y = savgol_filter(Y, 21, 2, axis=1)


    return Y

# FUNCTION egm_notch_filter
def egm_notch_filter(X, fs=1000, f_notch=50, f_quality=30, select_axis=1, harmonics=False, apply_times=1):

    # Create an n-order lowpass butterworth filter.
    if harmonics:
        b, a = ss.iircomb(w0=f_notch, Q=f_quality, ftype='notch', fs=fs)
    else:
        b, a = ss.iirnotch(w0=f_notch, Q=f_quality, fs=fs)
    # Use filtfilt to apply the filter.
    Y = X.copy()
    for n in range(apply_times):
        Y = ss.filtfilt(b, a, Y, axis=select_axis)


    return Y

# FUNCTION getBipolarElectrodeIndices
def getBipolarElectrodeIndices(catheter_model='PentaRay'):
    # Catheter models ['PentaRay', 'Lasso', 'Achieve', 'HDGrid']
    # CS catheter models ['Tetra', 'Deca', 'C', 'D']
    # Ablation catheter models ['SMARTTOUCH', 'B', 'C', 'D']

    if catheter_model == 'Deca':
        bip_indices_neg = np.array([1, 3, 5, 7, 9]) - 1
        bip_indices_pos = np.array([2, 4, 6, 8, 10]) - 1

    if catheter_model == 'Tetra':
        bip_indices_neg = np.array([1, 3]) - 1
        bip_indices_pos = np.array([2, 4]) - 1

    if catheter_model == 'SMARTTOUCH':
        bip_indices_neg = np.array([1, 3]) - 1
        bip_indices_pos = np.array([2, 4]) - 1

    if catheter_model == 'PentaRay':
        bip_indices_neg = np.array([1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]) - 1
        bip_indices_pos = bip_indices_neg + 1

    if catheter_model == 'Lasso':
        bip_indices_neg = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) - 1
        bip_indices_pos = bip_indices_neg + 1

    if catheter_model == 'Achieve':
        bip_indices_neg = np.array([1, 2, 3, 4, 5, 6, 7, 8]) - 1
        bip_indices_pos = bip_indices_neg + 1

    if catheter_model == 'HDGrid':
        # A1(1) B1(5) C1(9)  D1(13)
        # A2(2) B2(6) C2(10) D2(14)
        # A3(3) B3(7) C3(11) D3(15)
        # A4(4) B4(8) C4(12) D4(16)
        bip_indices_pos = np.array([2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 5, 9, 13, 6, 10, 14, 7, 11, 15, 8, 12, 16]) - 1
        bip_indices_neg = np.array([1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]) - 1

    return bip_indices_pos, bip_indices_neg

# FUNCTION calculateBipolarEGMs
def calculateBipolarEGMs(EGMs, catheter_model='PentaRay'):

    # # Catheter models ['PentaRay', 'Lasso', 'Achieve', 'HDGrid']
    # # CS catheter models ['Tetra', 'Deca', 'C', 'D']
    # # Ablation catheter models ['SMARTTOUCH', 'B', 'C', 'D']
    [Nu, L] = EGMs.shape
    # Bipolar
    # _EGMs = []
    #
    # if catheter_model == 'Deca':
    #     bip_indices_neg = np.array([1, 3, 5, 7, 9]) - 1
    #     bip_indices_pos = np.array([2, 4, 6, 8, 10]) - 1
    #     Nb = len(bip_indices_neg)
    #
    # if catheter_model == 'Tetra':
    #     bip_indices_neg = np.array([1, 3]) - 1
    #     bip_indices_pos = np.array([2, 4]) - 1
    #     Nb = len(bip_indices_neg)
    #
    # if catheter_model == 'SMARTTOUCH':
    #     bip_indices_neg = np.array([1, 3]) - 1
    #     bip_indices_pos = np.array([2, 4]) - 1
    #     Nb = len(bip_indices_neg)
    #
    # if catheter_model == 'PentaRay':
    #     bip_indices_neg = np.array([1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]) - 1
    #     bip_indices_pos = bip_indices_neg + 1
    #     Nb = len(bip_indices_neg)
    #
    # if catheter_model == 'Lasso':
    #     bip_indices_neg = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) - 1
    #     bip_indices_pos = bip_indices_neg + 1
    #     Nb = len(bip_indices_neg)
    #
    # if catheter_model == 'Achieve':
    #     bip_indices_neg = np.array([1, 2, 3, 4, 5, 6, 7, 8]) - 1
    #     bip_indices_pos = bip_indices_neg + 1
    #     Nb = len(bip_indices_neg)
    #
    # if catheter_model == 'HDGrid':
    #     # A1(1) B1(5) C1(9)  D1(13)
    #     # A2(2) B2(6) C2(10) D2(14)
    #     # A3(3) B3(7) C3(11) D3(15)
    #     # A4(4) B4(8) C4(12) D4(16)
    #     bip_indices_pos = np.array([2,3,4,  6,7,8,  10,11,12,  14,15,16,    5,9,13,  6,10,14,  7,11,15,  8,12,16]) - 1
    #     bip_indices_neg = np.array([1,2,3,  5,6,7,   9,10,11,  13,14,15,    1,5,9,   2,6,10,   3,7,11,   4,8,12]) - 1

    bip_indices_pos, bip_indices_neg = getBipolarElectrodeIndices(catheter_model=catheter_model)
    Nb = len(bip_indices_neg)

    # Calculate the bipolar EGMS
    Bipolar_EGMs = np.zeros((Nb, L))
    for b in range(Nb):
        aux_pos = bip_indices_pos[b]
        aux_neg = bip_indices_neg[b]
        aux_bipolar = EGMs[aux_pos, :] - EGMs[aux_neg, :]
        Bipolar_EGMs[b, :] = aux_bipolar

    return Bipolar_EGMs

# FUNCTION calculateBipolarElectrodePositions
def calculateBipolarElectrodePositions(electrode_positions, catheter_model='PentaRay', verbose=False):

    bipolar_electrode_positions = []

    if catheter_model == 'PentaRay':

        bip_indices_pos, bip_indices_neg = getBipolarElectrodeIndices(catheter_model=catheter_model)
        Nb = len(bip_indices_neg)

        # Calculate the bipolar electrode positions
        bipolar_electrode_positions = np.zeros((Nb, 3))
        for b in range(Nb):
            aux_pos = bip_indices_pos[b]
            aux_neg = bip_indices_neg[b]
            aux_bipolar_position = electrode_positions[aux_pos, :] / 2 + electrode_positions[aux_neg, :] / 2
            bipolar_electrode_positions[b, :] = aux_bipolar_position

    return bipolar_electrode_positions

# FUNCTION robustUnipolarEGMLATdetection
def robustUnipolarEGMLATdetection(Bipolar_egms, Unipolar_egms, Bipolar_relative_indices, Unipolar_relative_indices, LATSettings_bipolar, LATSettings_unipolar, show_figures, verbose=False, Binary_Activity=[]):
    """
    Function: robustUnipolarEGMLATdetection(Unipolar_egms, Bipolar_egms, Bipolar_relative_indices, Unipolar_relative_indices, LATSettings_unipolar, LATSettings_bipolar, show_figures, verbose)

    Parameters:
        Bipolar_egms (numpy float array): Bipolar EGM signals. Size [Nb, L]
        Unipolar_egms (numpy float array): Unipolar EGM signals. Size [Nu, L]
        Bipolar_relative_indices (int list): Bipolar index for each calculated bipolar EGM
        Unipolar_relative_indices (numpy int array): Unipolar electrode indices for each calculated bipolar signal. Size [Nb, 2]
        LATSettings_bipolar (LATSettingsClass): LAT detection settings for the unipolar signals
        LATSettings_unipolar (LATSettingsClass): LAT detection settings for the unipolar signals
        show_figures (bool): Plot LAT detection figure
        verbose (bool): Print additional information
        Binary_Activity (numpy float array): Bipolar binary activity signal calculated from NLEO. Size [Nb, L]

    Returns:
        Robust_results (List): Returns the robust detection as a list of length Nb
            Each list element contains = Robust_results[i] = [Robust_bipolar_LATs, Robust_bipolar_intervals, Dominant_cycle]
                - Robust_bipolar_LATs (list of numpy float array): List of length Nb containing the bipolar LATs
                - Robust_bipolar_intervals (list of numpy float array): List of length Nb containing the interval durations for the bipolar LATs
                - Dominant_cycle (list of numpy float array): List of the dominant cycle for each bipolar EGM
    Version:
        2024.07.25 - Added Binary_Activity filter to clean variables based on NLEO activity
    """
    # Return variables
    Robust_bipolar_LATs = []
    Robust_bipolar_intervals = []
    Dominant_cycle = []

    # Variables
    # Nb = len(Bipolar_relative_indices)
    Nb = Bipolar_egms.shape[0]
    [Nu, L] = Unipolar_egms.shape
    M_unipolar = LATSettings_unipolar.M
    M_bipolar = LATSettings_bipolar.M

    num_cores = multiprocessing.cpu_count()

    # Calculate all unipolar LATs
    if verbose:
        print('- Num cores: ' + str(num_cores))
        aux_print = '- Calculating unipolar LATS in parallel...'
        print(aux_print)
    # Unipolar beta
    aux_beta_unipolar = calculateUnipolarEGMSlope(Unipolar_egms, M_unipolar)
    aux_beta_unipolar_abs = -1 * aux_beta_unipolar
    aux_beta_unipolar_abs[aux_beta_unipolar_abs < 0] = 0
    Unipolar_lat_detection = Parallel(n_jobs=num_cores, verbose=0)(delayed(detectLATs)(aux_beta_unipolar_abs[u, :], LATSettings_unipolar) for u in range(Nu))

    # Calculate all bipolar LATs
    if verbose:
        aux_print = '- Calculating bipolar LATS in parallel...'
        print(aux_print)
    # Bipolar beta
    aux_beta_bipolar = calculateUnipolarEGMSlope(Bipolar_egms, M_bipolar)
    aux_beta_bipolar_abs = np.abs(aux_beta_bipolar)

    # print('adasdaggggg¡?=')
    # print(Nb)
    # print(aux_beta_bipolar_abs.shape)

    Bipolar_lat_detection = Parallel(n_jobs=num_cores, verbose=0)(delayed(detectLATs)(aux_beta_bipolar_abs[b, :], LATSettings_bipolar) for b in range(Nb))


    # TO BE DONE.. parallelize this loop...
    # Call auxiliary function...
    Robust_results = Parallel(n_jobs=num_cores, verbose=0)(delayed(robustLATAuxiliaryFunction)(b, L, Bipolar_lat_detection[b], Unipolar_relative_indices, Unipolar_lat_detection, verbose) for b in range(Nb))
    # Robust_results[i] = [Robust_bipolar_LATs, Robust_bipolar_intervals, Dominant_cycle]
    #for b in range(Nb):
     #   print(b)
      #  R = robustLATAuxiliaryFunction(b, L, Bipolar_lat_detection[b], Unipolar_relative_indices, Unipolar_lat_detection, verbose=True)
    #b, L, bipolar_LATDetection, Unipolar_relative_indices, Unipolar_lat_detection, verbose


    if Binary_Activity != []:
        # Binary activity filter. If the activity is low (0), the LATs are not considered
        if verbose:
            aux_print = '- Cleaning LATs based on binary activity...'
            print(aux_print)

        for b in range(Nb):
            if verbose:
                aux_print = '  - Bipolar signal: ' + str(b+1)
                print(aux_print)
            aux_binary_activity = Binary_Activity[b, :]
            aux_bipolar_lats = Robust_results[b][0].astype(int)
            aux_lat_bin_activity = aux_binary_activity[aux_bipolar_lats]
            aux_valid = np.where(aux_lat_bin_activity > 0)[0]
            aux_bipolar_lats_clean = aux_bipolar_lats[aux_valid]

            aux_bipolar_intervals = np.diff(aux_bipolar_lats_clean)
            if len(aux_bipolar_intervals)>0:
                aux_dominant_cycle = stats.mode(aux_bipolar_intervals)[0][0]
            else:
                aux_dominant_cycle = []

            # Store the clean LATs, first create the tupple and then assign it to the list
            aux_clean_result = [aux_bipolar_lats_clean, aux_bipolar_intervals, aux_dominant_cycle]
            Robust_results[b] = aux_clean_result

    # print(Robust_results)
    return Robust_results

# FUNCTION calculateUnipolarEGMSlopeParallel
def calculateUnipolarEGMSlopeParallel(egm_signals, m):
    """
    Function: calculateUnipolarEGMSlopeParallel(egm_signals, m)

    Parameters:
        egm_signals (numpy float array): Bipolar EGM signals. Size [Nu, L] or [L,]
        m (int) : Slope approximation filter order
    Returns:
        beta (numply float array): Slope filter result. Size equals to the egm_signals size
    """
    return calculateUnipolarEGMSlopeVectorized(egm_signals, m)

# FUNCTION calculateUnipolarEGMSlopeVectorized
def calculateUnipolarEGMSlopeVectorized(egm_signals, m):
    """
    Function: calculateUnipolarEGMSlopeVectorized(egm_signals, m)

    Vectorized slope calculation using scipy.ndimage.convolve1d.
    Replaces the per-signal joblib parallelism with a single vectorized operation.

    Parameters:
        egm_signals (numpy float array): EGM signals. Size [Nu, L] or [L,]
        m (int) : Slope approximation filter order
    Returns:
        beta (numpy float array): Slope filter result. Size equals to the egm_signals size
    """
    from scipy.ndimage import convolve1d

    squeeze = False
    if egm_signals.ndim == 1:
        egm_signals = egm_signals.reshape(1, -1)
        squeeze = True

    h = np.linspace(-m, m, 2 * m + 1)
    B = np.sum(h ** 2)
    kernel = h / B

    beta = convolve1d(egm_signals.astype(np.float64), kernel, axis=1, mode='constant', cval=0.0)

    if squeeze:
        beta = beta.squeeze(axis=0)

    return beta

# FUNCTION calculateUnipolarEGMSlope
def calculateUnipolarEGMSlope(egm_signals, m):
    """
    Function: calculateUnipolarEGMSlope(egm_signals, m)

    Parameters:
        egm_signals (numpy float array): Bipolar EGM signals. Size [Nu, L] or [L,]
        m (int) : Slope approximation filter order
    Returns:
        beta (numply float array): Slope filter result. Size equals to the egm_signals size
    """
    aux_dim = egm_signals.ndim
    # Check input signal dimensions
    if aux_dim == 1:
        Nu = 1
        L = len(egm_signals)
    else:
        [Nu, L] = egm_signals.shape

    # Filter coefficients
    h = np.linspace(-m, m, 2 * m + 1)
    # Constants
    aux_m_range = np.linspace(-m, m, 2*m+1)
    aux_m_range = np.power(aux_m_range, 2)
    aux_B = np.sum(aux_m_range)

    beta = np.zeros(egm_signals.shape)

    for n in range(L):

        aux_min_index = n - m
        aux_max_index = n + m

        # Check indices limits
        if aux_min_index < 0:
            aux_min_index = 0
        if aux_max_index > L-1:
            aux_max_index = L-1
        # Temporal window indices
        aux_indices = np.linspace(aux_min_index, aux_max_index, abs(aux_max_index - aux_min_index) + 1).astype(int)
        aux_h = h[aux_indices + m - n] * (1/aux_B)

        for s in range(Nu):
            # Process all windows
            if Nu == 1:
                aux_signal = egm_signals[aux_indices]
            else:
                aux_signal = egm_signals[s, aux_indices]

            aux_beta = np.multiply(aux_signal, aux_h)
            aux_beta = np.sum(aux_beta)

            if Nu == 1:
                beta[n] = aux_beta
            else:
                beta[s, n] = aux_beta
    return beta

# FUNCTION detectLATs()
def detectLATs(egm_signal, LATSettings):
    """
    Function: calculateUnipolarEGMSlopeParallel(egm_signals, LATSettings)

    Parameters:
        egm_signal (numpy float array): EGM signals. Size [L,]
        LATSettings (LATSettingsClass) : Settings to perform the LAT detection
    Raises:
            ValueError: Signal is not 1D
    Returns:
        LATDetection (LATDetectionClass) : containing the LAT indices, EGM values at LATs
    """
    # LAT Settings
    fs = LATSettings.fs
    tau_input = LATSettings.tau_input
    blank_period_time = LATSettings.blank_period_time
    T_prev = LATSettings.T_prev
    sigma_abs_th = LATSettings.sigma_abs_th
    LATs_search = LATSettings.LATs_search

    # Signal dimensions
    aux_dim = egm_signal.ndim
    # Check input signal dimensions
    if aux_dim == 1:
        L = len(egm_signal)
    else:
        # Raise the error
        aux_error_str = 'ERROR: only 1D signals supported'
        raise ValueError(aux_error_str)

    # Skip first T_prev seconds
    t0 = int(T_prev*fs)
    if t0 < 1:
        t0 = 1
    # End time
    tf = L-2

    # Threshold
    new_Mi = sigma_abs_th
    threshold = np.zeros(egm_signal.shape)
    abs_threshold = np.zeros(egm_signal.shape)
    isocro = np.zeros(egm_signal.shape)
    isocro[0] = -100

    # Exponential decay variable
    tau = tau_input * fs
    # Blank period in samples
    blank_period = blank_period_time * fs
    # Amplitude of the previously detected peak. Initialized as the maximum value in the signal
    Mi = np.max(egm_signal)
    # Local search window span
    local_ti = -blank_period/2
    sigma_t = 0

    # LATs
    activation_peaks_values = np.zeros(egm_signal.shape)
    activation_peaks_indices_list = []

    end_blank_period = 0
    rest_return = 1
    add_lat = 0

    new_t = -1

    # Pre-extract signal values to avoid repeated numpy indexing overhead
    sig = egm_signal
    local_ti_plus_bp = local_ti + blank_period
    half_blank = blank_period / 2
    bp_1_2 = 1.2 * blank_period

    for t in range(t0, tf + 1):

        # Blank period
        isocro[t] = isocro[t-1] - 1

        if t > local_ti_plus_bp or (t == tf and add_lat == 1) or t < blank_period:

            if end_blank_period:

                add_lat = 0

                activation_peaks_values[new_t] = sig[new_t]
                activation_peaks_indices_list.append(new_t)

                Mi = new_Mi

                prev_t = np.arange(new_t, t + 1)
                threshold[prev_t] = Mi

                # LINEAR ISOCRO UPDATE
                n_prev = len(prev_t)
                isocro[prev_t] = np.linspace(0, -n_prev + 1, n_prev)

                end_blank_period = 0

            # MODIFIED
            sigma_t = sigma_abs_th

            mi_minus_sigma = Mi - sigma_t
            aux_exponent = -mi_minus_sigma * (t - local_ti_plus_bp) / tau

            aux_th = mi_minus_sigma * math.exp(aux_exponent) + sigma_t

            threshold[t] = aux_th

            if threshold[t] < sigma_abs_th:
                threshold[t] = sigma_abs_th

            if sig[t] >= threshold[t] and rest_return:

                # PEAK
                if sig[t] - sig[t+1] > 0:
                    add_lat = 1

                    local_ti = t
                    local_ti_plus_bp = t + blank_period
                    end_blank_period = 1

                    new_t = t
                    new_Mi = sig[t]

                    rest_return = 0

        else:

            # SEARCH FOR LOCAL MAX
            threshold[t] = threshold[t-1]

            if t < local_ti + half_blank:
                # New peak
                if sig[t] > threshold[t] and sig[t] - sig[t-1] > 0 and sig[t] > new_Mi:
                    new_Mi = sig[t]
                    new_t = t

                    local_ti = t
                    local_ti_plus_bp = t + blank_period

        # rest_return condition
        if (sig[t] == 0 and rest_return == 0) or (rest_return == 0 and t > local_ti + bp_1_2):
            rest_return = 1

        abs_threshold[t] = sigma_t

    isocro[L-1] = isocro[L-2] - 1

    activation_peaks_indices = np.array(activation_peaks_indices_list, dtype=float)

    LATDetection = LATDetectionClass(LATSettings, activation_peaks_values, activation_peaks_indices, threshold, abs_threshold, isocro)

    return LATDetection

def robustLATAuxiliaryFunction(b, L, bipolar_LATDetection, Unipolar_relative_indices, Unipolar_lat_detection, verbose):

    if verbose:
        aux_print = '- Bipolar signal: ' + str(b+1)
        print(aux_print)

    aux_unipolar_index1 = Unipolar_relative_indices[b, 0]
    aux_unipolar_index2 = Unipolar_relative_indices[b, 1]
    # print('indices: ' + str(aux_unipolar_index1), ', ' + str(aux_unipolar_index2))
    unipolar_LATDetection1 = Unipolar_lat_detection[aux_unipolar_index1]
    unipolar_LATDetection2 = Unipolar_lat_detection[aux_unipolar_index2]

    # CORRECT THE LATs
    # Use the bipolar LATs and correct them with the unipolar LAT information
    aux_unipolar_lats1 = unipolar_LATDetection1.activation_peaks_indices
    aux_unipolar_lats2 = unipolar_LATDetection2.activation_peaks_indices
    aux_bipolar_lats = bipolar_LATDetection.activation_peaks_indices
    num_bipolar_lats = len(aux_bipolar_lats)
    aux_lat_search = bipolar_LATDetection.LATSettings.LATs_search
    aux_corrected_bipolar_lats = np.copy(aux_bipolar_lats)
    for r in range(num_bipolar_lats):

        aux_lat = aux_bipolar_lats[r]
        aux_search_init = aux_lat - aux_lat_search
        aux_search_end = aux_lat + aux_lat_search
        if aux_search_init < 0:
            aux_search_init = 0
        if aux_search_end > L-1:
            aux_search_end = L-1
        aux_indices_len = int(aux_search_end-aux_search_init+1)
        aux_search_indices = np.linspace(aux_search_init, aux_search_end, aux_indices_len)
        # print(aux_search_indices)

        # Match is member
        aux_where1 = np.where(np.logical_and(aux_unipolar_lats1 > aux_search_indices[0], aux_unipolar_lats1<aux_search_indices[-1]))
        aux_where2 = np.where(np.logical_and(aux_unipolar_lats2 > aux_search_indices[0], aux_unipolar_lats2 < aux_search_indices[-1]))
        aux_lats_reduced1 = aux_unipolar_lats1[aux_where1]
        aux_lats_reduced2 = aux_unipolar_lats2[aux_where2]
        aux_corrected_unipolar_lat = -1
        # If both unipolar LATs exist take the mean value
        if len(aux_lats_reduced1) + len(aux_lats_reduced2) > 1:
            if len(aux_lats_reduced1) > 0: # just in case...
                if len(aux_lats_reduced2) > 0: # just in case...
                # print('aaaaaaaaaaaa')
                # print(aux_lats_reduced1)
                # print(aux_lats_reduced2)
                    aux_corrected_unipolar_lat = int((aux_lats_reduced1[0]+aux_lats_reduced2[0])/2)
        else:
            if len(aux_lats_reduced1) > 0:
                aux_corrected_unipolar_lat = aux_lats_reduced1[0]
            if len(aux_lats_reduced2) > 0:
                aux_corrected_unipolar_lat = aux_lats_reduced2[0]

        # If unipolar LAT in the temporal window of interest
        if aux_corrected_unipolar_lat > -1:
            if verbose:
                aux_print = ' - Bipolar LAT: ' + str(aux_lat) + ', corrected to ' + str(aux_corrected_unipolar_lat)
                print(aux_print)
            aux_corrected_bipolar_lats[r] = aux_corrected_unipolar_lat

    # Add possible unipolar LATs that might not be detected due to low bipolar EGM voltage
    search_correction = 0.5
    aux_lat_diff = np.diff(aux_corrected_bipolar_lats)

    if len(aux_lat_diff) > 1:

        aux_lat_mode = stats.mode(aux_lat_diff)[0][0]
        aux_lat_search_add = int(np.ceil(aux_lat_mode*search_correction))

        # aux_new_lats = np.empty((0,0))
        for u in range(2):
            if u == 0:
                aux_lats = aux_unipolar_lats1
            else:
                aux_lats = aux_unipolar_lats2
            for i in range(len(aux_lats)):

                aux_lat_u = aux_lats[i]

                aux_search_init = aux_lat_u - aux_lat_search_add
                aux_search_end = aux_lat_u + aux_lat_search_add
                if aux_search_init < 0:
                    aux_search_init = 0
                if aux_search_end > L - 1:
                    aux_search_end = L - 1
                aux_indices_len = int(aux_search_end - aux_search_init + 1)
                aux_search_indices = np.linspace(aux_search_init, aux_search_end, aux_indices_len)

                aux_where = np.where(np.logical_and(aux_corrected_bipolar_lats > aux_search_indices[0], aux_corrected_bipolar_lats<aux_search_indices[-1]))
                if len(aux_where) == 0:
                    # There is no bipolar LAT, include the new unipolar LAT!
                    aux_corrected_bipolar_lats = np.appen(aux_corrected_bipolar_lats, aux_where[0])
                    aux_corrected_bipolar_lats = np.sort(aux_corrected_bipolar_lats)

                    if verbose:
                        aux_print = '  - New missed unipolar LAT added!'
                        print(aux_print)

    if verbose:
        aux_print = '- Bipolar LATs: ' + str(aux_bipolar_lats)
        print(aux_print)
        aux_print = ' Bipolar LATs corrected: ' + str(aux_corrected_bipolar_lats)
        print(aux_print)

    # # Store corrected LATs
    # Robust_bipolar_LATs.append(aux_corrected_bipolar_lats)
    #
    # # Calculate LAT intervals with the corrected bipolar LATs
    aux_intervals = np.diff(aux_corrected_bipolar_lats)
    # Robust_bipolar_intervals.append(aux_intervals)
    #
    if len(aux_intervals) == 0:
        aux_dominant_cycle = 0
    else:
        aux_dominant_cycle = stats.mode(aux_intervals)[0][0]
    # Dominant_cycle.append(aux_dominant_cycle)
    return aux_corrected_bipolar_lats, aux_intervals, aux_dominant_cycle
    
# FUNCTION loadSettings (RAISApp)
def loadSettings(ini_file):
    ini_config = configparser.ConfigParser()
    ini_config.read(ini_file)
    return ini_config



def getElectrodesRingOrder(numsensor, num_legs, numsensorperleg, typeofpol='Unipolar', catheter_model='PentaRay'):

    if catheter_model == 'OctaRay':
        vector = np.linspace(0, numsensor - 1 -4, num=numsensor-4)
        if typeofpol == 'Unipolar':
            value = [32, 34, 33, 35] # Unipolar
        else:
            value = [24, 26, 25, 27] # Bipolar
    else:
        vector = np.linspace(0, numsensor - 1, num=numsensor)

    vector = vector.astype(int)
    matrix = np.reshape(vector, (num_legs, numsensorperleg))
    newmatrix = np.transpose(matrix)
    newvector = newmatrix.ravel()

    if catheter_model == 'OctaRay':
        newvector = np.append(newvector, value)

    return newvector







def calculateBipolarSignalsFromBranchesPentaRay(Unipolar_signals, CatheterPosition_Branch, Electrode_positions, verbose):
    """
    Function: calculateBipolarSignals(Unipolar_signals, CatheterPosition_Branch, Electrode_positions, verbose)

    Parameters:
        Unipolar_signals (numpy float array): Unipolar EGM signals. Size [Nu, L]
        CatheterPosition_Branch (numpy float array): Electrode position with respect the branch it belong to. 1-4 for the PentaRay
        Electrode_positions (numpy float array): Unipolar electrode positions. Size [Nu, 3]
        verbose (bool): Print additional information

    Returns:
        Bipolar_signals (numpy float array): Bipolar EGM signals. Size [Nb, L]
        Bipolar_Electrode_positions (numpy float array): Bipolar electrode positions. Size [Nb, 3]
        Bipolar_relative_indices (int list): Bipolar index for each calculated bipolar EGM
        Unipolar_relative_indices (numpy int array): Unipolar electrode indices for each calculated bipolar signal. Size [Nb, 2]
        Unipolar_signals_output (numpy float array): Unipolar signals. Size [20, L] in case some electrodes are missing they are filled with zeros
        Unipolar_indices (numpy int array): Unipolar indices for all the 15 bipolar signals
        Unipolar_Electrode_positions_ouput (numpy float array): Bipolar electrode positions. Size [Nu, 3] with NaN values if missing electrodes
   """
    # CatheterPosition_Branch should be:
    #   CatheterPosition_Branch = [1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5];
    # JUANCOMENTA: va de electrodo en electrodo, y los numeros indican a q brazo del pentarray pertenecen
    Unipolar_indices = np.array(([0, 1], [1, 2], [2, 3],
                                [4, 5], [5, 6],[6, 7],
                                [8, 9], [9, 10], [10, 11],
                                [12, 13], [13, 14], [14, 15],
                                [16, 17], [17, 18], [18, 19])
                                )
    Unipolar_indices = Unipolar_indices.astype(int)
    # Number of unipolar signals and signal length
    [Nu, L] = Unipolar_signals.shape
    Unipolar_signals_output = np.zeros((20, L))
    # Number of bipolar signals (PentaRay catheter)
    Nb = 15 #este numero es por el numero de parejas de los electrodos: 5x3
    num_bipolar_egms_per_branch = 3
    num_unipolar_egms_per_branch = 4

    if verbose:
        aux_print = 'Calculating bipolar EGMs from unipolar EGMs (PentaRay catheter)'
        print(aux_print)
        aux_print = '- Number of unipolar signals: ' + str(Nu)
        print(aux_print)
        aux_print = '- Signal length: ' + str(L) + ' ms'
        print(aux_print)
        aux_print = '- Number of bipolar signals: ' + str(Nb)
        print(aux_print)

    # Return variables
    Bipolar_signals = np.zeros((Nb, L))
    Bipolar_Electrode_positions = np.empty((Nb, 3))
    Bipolar_Electrode_positions[:] = np.nan
    Bipolar_relative_indices = []
    Unipolar_relative_indices = []
    Unipolar_Electrode_positions_ouput = np.empty((20, 3))
    Unipolar_Electrode_positions_ouput[:] = np.nan

    Bipolar_interpolated_positions = np.empty((Nb, 3))
    Bipolar_interpolated_positions[:] = np.nan

    aux_branch_counter = 0
    aux_branch_counter_unipolar = 0
    aux_last_branch = -1
    for b in range(Nu):
        aux_new_branch = CatheterPosition_Branch[b]

        if aux_last_branch == aux_new_branch:
            aux_branch_counter_unipolar = aux_branch_counter_unipolar + 1
        else:
            aux_branch_counter_unipolar = 0

        # Unipolar egm
        aux_unipolar_index = (aux_new_branch - 1) * num_unipolar_egms_per_branch + aux_branch_counter_unipolar  # 0-19
        aux_unipolar_index = aux_unipolar_index.astype(int)

        Unipolar_signals_output[aux_unipolar_index, :] = Unipolar_signals[b, :]
        Unipolar_Electrode_positions_ouput[aux_unipolar_index, : ] = Electrode_positions[b, :]

        # print(b, aux_unipolar_index, Electrode_positions)

        if aux_last_branch == aux_new_branch:
            # Calculate bipolar
            aux_e2 = b
            aux_e1 = b-1
            aux_u1 = Unipolar_signals[aux_e1, :]
            aux_u2 = Unipolar_signals[aux_e2, :]

            # Bipolar egm
            aux_bipolar_index = (aux_new_branch-1)*num_bipolar_egms_per_branch + aux_branch_counter - 1  # 0-14
            aux_bipolar_index = aux_bipolar_index.astype(int)
            aux_bipolar_signal = aux_u2 - aux_u1

            if verbose:
                aux_print = ' - Bipolar signal: ' + str(aux_bipolar_index+1) + '/' + str(Nb)
                print(aux_print)
                aux_print = '  - Unipolar E2 index: ' + str(aux_e2) + ' (' + str(aux_e2+1) + ')'
                print(aux_print)
                aux_print = '  - Unipolar E1 index: ' + str(aux_e1) + ' (' + str(aux_e1+1) + ')'
                print(aux_print)

            Bipolar_signals[aux_bipolar_index, :] = aux_bipolar_signal
            # Bipolar electrode position
            aux_position = Electrode_positions[aux_e2, :]
            Bipolar_Electrode_positions[aux_bipolar_index, :] = aux_position #JUANCOMENTA: sets as second unipolar position
            # More precise to use the middle point of the two signals
            Bipolar_interpolated_positions[aux_bipolar_index] = 0.5*(Electrode_positions[aux_e2, :] + Electrode_positions[aux_e1, :])

            aux_branch_counter = aux_branch_counter + 1
            Bipolar_relative_indices.append(aux_bipolar_index)
            # aux_u = np.array([aux_e1, aux_e2])
            aux_u = np.array([aux_unipolar_index-1, aux_unipolar_index])
            # if aux_bipolar_index == 0:
            if len(Unipolar_relative_indices) == 0:
                Unipolar_relative_indices = aux_u.reshape((1, 2))
            else:
                Unipolar_relative_indices = np.vstack((Unipolar_relative_indices, aux_u))
        else:
            # Different branch
            aux_branch_counter = 1

        aux_last_branch = aux_new_branch

    Bipolar_Electrode_positions = Bipolar_interpolated_positions

    return Bipolar_signals, Bipolar_Electrode_positions, Bipolar_relative_indices, Unipolar_relative_indices, Unipolar_signals_output, Unipolar_indices, Unipolar_Electrode_positions_ouput


def correctCartoFinderElectrodesRAcFAc(Nu, CatheterPosition_Branch, rac_score, fac_score):

    rac_score_corrected = np.zeros((Nu,))
    fac_score_corrected = np.zeros((Nu,))
    for n in range(Nu):
        rac_score_corrected[n] = np.nan
        fac_score_corrected[n] = np.nan

    # aux_CatheterPosition_Branch = aux_patient_cartofinderdata.CatheterPosition_Branch

    # Function
    num_unipolar_egms_per_branch = 4
    num_electrodes = len(CatheterPosition_Branch)

    aux_branch_counter = 0
    aux_branch_counter_unipolar = 0
    aux_last_branch = -1
    for b, aux_new_branch in enumerate(CatheterPosition_Branch):

        if aux_last_branch == aux_new_branch:
            aux_branch_counter_unipolar = aux_branch_counter_unipolar + 1
        else:
            aux_branch_counter_unipolar = 0

        # Unipolar egm
        aux_unipolar_index = (aux_new_branch - 1) * num_unipolar_egms_per_branch + aux_branch_counter_unipolar  # 0-19
        aux_unipolar_index = aux_unipolar_index.astype(int)

        rac_score_corrected[aux_unipolar_index] = rac_score[b]
        fac_score_corrected[aux_unipolar_index] = fac_score[b]

        if aux_last_branch == aux_new_branch:
            aux_branch_counter = aux_branch_counter + 1
        else:
            aux_branch_counter = 1

        aux_last_branch = aux_new_branch

    return rac_score_corrected, fac_score_corrected