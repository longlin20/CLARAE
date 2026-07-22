# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 17:14:09 2019

@author: griosm
"""
import matplotlib.pyplot as plt
import numpy as np
# import scipy.ndimage as ndi
from scipy import signal
from scipy.signal import windows
from scipy.fftpack import fft, fftshift
# from sklearn.externals import joblib
# from scipy.ndimage import gaussian_filter1d
# from HMM_functions import *
# scikit-dsp-comm
# https://scikit-dsp-comm.readthedocs.io/en/latest/nb_examples/FIR_and_IIR_Filter_Design.html
import sk_dsp_comm.fir_design_helper as fir_d
from tqdm import tqdm


# from postProcessingFunctions import *


def calculateNLEO(Bipolar_signals, NLEO_threshold=0.1, window_len=50, sigma_value=7, Merge_distance=42,
                  Outlier_distance=0, Min_NLEO=0.0001, Max_nleo_std=0.01, Verbose=False, SHOW_FIGURES=False):
    # [NLEO, NLEO_Activity] = calculateNLEO(Bipolar_signals)

    N, L = Bipolar_signals.shape

    NLEO = np.zeros((N, L))
    NLEO_LPF = np.zeros((N, L))
    NLEO_Binary_Activity = np.zeros((N, L))
    NLEO_Activity = np.zeros((N, 1))

    ## PAPER BY Schilling 2009
    # Non-Linear Energy Operator for the Analysis of Intracardial Electrograms
    # Steps
    # 1. Denoising and Baseline Wander Removal
    # 2. NLEO
    # 3. Gaussian Lowpass Filtering
    # 4. Adaptive Thresholding: recommended 0.1 times the std(NLEO signal) = NLEO_threshold * np.str(NLEO)
    # 5. Postprocessing --> < 42 ms of blank period between activations is considered active

    ## 1. BPF to remove the low freq components is not necessary if using Carto signals
    # --> SKIP
    # Equiripple LPF f_Stop at 1/8 of fs 150Hz for 1200Hz, and attenuation -32dB
    #    b_k = fir_d.firwin_kaiser_lpf(1/8,1/6,50,1.0)
    # FIR No feedback a = 1
    #    b_r = fir_d.fir_remez_lpf(1/8,1/6,0.2,50,1.0)
    b_r = fir_d.fir_remez_lpf(125, 150, 0.5, 32, 1000)

    if SHOW_FIGURES:
        plt.figure()
        fir_d.freqz_resp_list([b_r, b_r], [[1], [1]], 'dB', fs=1)
        plt.ylim([-80, 5])
        plt.title(r'Equal Ripple Lowpass')
        plt.ylabel(r'Filter Gain (dB)')
        plt.xlabel(r'Frequency in kHz')
        plt.grid()

    #    plt.figure()
    #    fir_d.freqz_resp_list([b_k,b_r],[[1],[1]],'dB',fs=1)
    #    plt.ylim([-80,5])
    #    plt.title(r'Kaiser vs Equal Ripple Lowpass')
    #    plt.ylabel(r'Filter Gain (dB)')
    #    plt.xlabel(r'Frequency in kHz')
    #    plt.legend((r'Kaiser: %d taps' % len(b_k),r'Remez: %d taps' % len(b_r)),loc='best')
    #    plt.grid()
    #    plt.show()

    ## 2. NLEO
    for n in tqdm(range(N), desc="  NLEO", disable=(N < 50)):
        aux_bipolar = Bipolar_signals[n, :]

        aux_filtered_bipolar = np.convolve(aux_bipolar, b_r, mode='same')

        aux_nleo = np.zeros((L,))
        aux_binary_activity = np.zeros((L,))

        for i in range(L):
            if i == 0:
                aux_prev = aux_filtered_bipolar[i]
            else:
                aux_prev = aux_filtered_bipolar[i - 1]

            if i == L - 1:
                aux_next = aux_filtered_bipolar[i]
            else:
                aux_next = aux_filtered_bipolar[i + 1]

            aux_nleo[i] = aux_filtered_bipolar[i] * aux_filtered_bipolar[i] - aux_prev * aux_next

        NLEO[n, :] = aux_nleo
        #        aux_nleo = np.reshape(aux_nleo, (L,))
        ## 3. Gaussian Lowpass Filtering
        # Other windows...
        #        window_type = 'blackman'
        #        aux_window = eval('np.'+ window_type +'(window_len)')
        #        aux_smooth_nleo = np.convolve(aux_nleo, aux_window/aux_window.sum(), mode='same')
        # Gaussian 1D kernel
        aux_gaussian_window = windows.gaussian(window_len, std=sigma_value)
        aux_smooth_nleo = np.convolve(aux_nleo, aux_gaussian_window, mode='same')

        aux_low_nleo_indices = np.where(aux_smooth_nleo < Min_NLEO)
        aux_smooth_nleo[aux_low_nleo_indices] = 0

        NLEO_LPF[n, :] = aux_smooth_nleo

        ## 4. Adaptive Thresholding wrt STD(NLEO)
        aux_nleo_std = np.std(aux_nleo)

        if aux_nleo_std > Max_nleo_std:
            aux_nleo_std = Max_nleo_std
        aux_threshold = NLEO_threshold * aux_nleo_std

        aux_indices = np.where(aux_smooth_nleo > aux_threshold)
        aux_binary_activity[aux_indices] = 1

        ## 5. Postprocessing
        #        [Activity_merged, Binary_Activity_merged] = mergeBinaryActivity(Binary_Activity, Merge_distance, Verbose)
        aux_binary_activity = np.reshape(aux_binary_activity, (1, L))
        [aux_activity_merged, aux_binary_activity_merged] = mergeBinaryActivity(aux_binary_activity, Merge_distance,
                                                                                Verbose)

        # Discard outliers
        [aux_activity_merged_outliers, aux_binary_activity_merged_outliers] = discardOutliersBinaryActivity(
            aux_binary_activity_merged, Outlier_distance, Verbose)

        NLEO_Binary_Activity[n, :] = aux_binary_activity_merged_outliers

        #        NLEO_Activity[n] = len(aux_indices[0])/L
        NLEO_Activity[n] = aux_activity_merged_outliers

        if SHOW_FIGURES:
            plt.figure()
            plt.stem(aux_gaussian_window)
            aux_title = 'Gaussian LPF impulse response'
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            plt.title(aux_title)

            aux_zoom = 10
            plt.figure()
            aux_label = 'Bipolar EGM'
            plt.plot(aux_bipolar.T, label=aux_label)
            aux_label = 'LPF EGM'
            plt.plot(aux_filtered_bipolar.T - 1, label=aux_label)
            aux_label = 'NLEO'
            plt.plot(aux_zoom * aux_nleo.T - 2, label=aux_label)
            aux_label = 'Gaussian LPF NLEO'
            plt.plot(aux_smooth_nleo.T - 3, label=aux_label)
            aux_label = 'Activity detection threshold'
            plt.plot([0, L - 1], [aux_threshold - 3, aux_threshold - 3], '--', label=aux_label)
            aux_label = 'Binary Activity'
            plt.plot(aux_binary_activity.T - 4, label=aux_label)
            aux_label = 'Postprocessed Binary Activity'
            plt.plot(aux_activity_merged_outliers.T - 5, label=aux_label)
            plt.legend()
            plt.show()
            aux_title = 'AR = ' + str(np.sum(aux_binary_activity) / L) + ' %, ' + 'AR_post = ' + str(
                aux_activity_merged_outliers[0]) + ' %'
            plt.title(aux_title)

    return NLEO, NLEO_LPF, NLEO_Binary_Activity, NLEO_Activity


###############################################################################
# Function call:
#    Activity_merged, Binary_Activity_merged = mergeBinaryActivity(Binary_Activity, Merge_distance, Verbose)
# Inputs:
#    Binary_Activity: [N, L] binary signal, N number of signals, L length
#    Merge_distance: maximum period of inactivity between two active signals to be set to active
#    Verbose: display/print additional information
# Outputs:
#    Activity_merged: [N,1] signal containing the Activity ratio (% Active samples/total time)
#    Binary_Activity_merged: [N,L] binary signal with the new merged binary active/inactive signals
# Function description:
#    This function merges binary activity signals that are active based on the
#    distance between activation segments. If there is an inactive period (signal=0)
#    smaller than the value Merge_distance between two active periods (signal=1),
#    the inactive period is set to ACTIVE
###############################################################################
def mergeBinaryActivity(Binary_Activity, Merge_distance, Verbose=False):
    [Nu, L] = Binary_Activity.shape

    Binary_Activity_merged = Binary_Activity.copy()
    Activity_merged = np.zeros((Nu,))

    for n in range(Nu):

        aux_binary_activity = Binary_Activity_merged[n, :]
        aux_last_value = 0
        aux_last_value_t = 0

        for t in range(L):

            aux_current_value = aux_binary_activity[t]

            if aux_current_value > aux_last_value:

                # new activation at t!
                if t > aux_last_value_t:

                    aux_elapsed_time = t - aux_last_value_t
                    if aux_elapsed_time < Merge_distance:
                        Binary_Activity_merged[n, aux_last_value_t:t] = 1
            if aux_current_value < aux_last_value:
                # Activation ends
                aux_last_value_t = t

            aux_last_value = aux_current_value

        aux_count = np.sum(Binary_Activity_merged[n, :])
        Activity_merged[n] = aux_count / L

    return Activity_merged, Binary_Activity_merged


###############################################################################

###############################################################################

def discardOutliersBinaryActivity(Binary_Activity_merged, Outlier_distance=0, Verbose=False):
    [N, L] = Binary_Activity_merged.shape

    Binary_Activity_merged_outliers = np.zeros(Binary_Activity_merged.shape)  # Binary_Activity_merged.copy()
    Activity_merged_outliers = np.zeros((N,))

    for n in range(N):

        aux_binary_activity = Binary_Activity_merged[n, :]
        Binary_Activity_merged_outliers[n, :] = aux_binary_activity
        aux_last_value = 0

        for t in range(L):

            aux_current_value = aux_binary_activity[t]

            if aux_current_value > aux_last_value:

                # new activation at t!
                if Verbose:
                    aux_print = 'New activation at t=' + str(t)
                    print(aux_print)
                aux_start = t
                aux_last_value = aux_current_value

            if aux_current_value < aux_last_value or (aux_last_value == 1 and t == L - 1):
                # Activation ends
                aux_last_value = aux_current_value
                aux_end = t
                aux_elapsed_time = aux_end - aux_start

                if Verbose:
                    aux_print = 'Activation ends at t=' + str(t)
                    print(aux_print)
                    aux_print = 'Activation duration =' + str(aux_elapsed_time)
                    print(aux_print)

                if aux_elapsed_time <= Outlier_distance:
                    Binary_Activity_merged_outliers[n, aux_start:aux_end] = 0

        aux_count = np.sum(Binary_Activity_merged_outliers[n, :])
        Activity_merged_outliers[n] = aux_count / L

    return Activity_merged_outliers, Binary_Activity_merged_outliers


###############################################################################


# %% OLD
def plotActivityEGMbyNLEO(Bipolar_signals, Binary_Activity_NLEO, Verbose):
    [Nu, L] = Bipolar_signals.shape

    plt.figure()
    aux_offset = 0
    aux_half_offset = 0.5

    for n in range(Nu):
        aux_egm = Bipolar_signals[n, :] + aux_offset + aux_half_offset
        #        aux_bin_activity = Binary_Activity[n,:] + aux_offset
        aux_bin_activity_nleo = Binary_Activity_NLEO[n, :] + aux_offset

        plt.plot(aux_egm)
        #        plt.plot(aux_bin_activity)
        plt.plot(aux_bin_activity_nleo, '--')

        aux_offset = aux_offset - 1


def plotActivityEGMbyHMMandNLEO(Bipolar_signals, Binary_Activity, Binary_Activity_NLEO, Verbose):
    [Nu, L] = Bipolar_signals.shape

    plt.figure()
    aux_offset = 0
    aux_half_offset = 0.5

    for n in range(Nu):
        aux_egm = Bipolar_signals[n, :] + aux_offset + aux_half_offset
        aux_bin_activity = Binary_Activity[n, :] + aux_offset
        aux_bin_activity_nleo = Binary_Activity_NLEO[n, :] + aux_offset

        plt.plot(aux_egm)
        plt.plot(aux_bin_activity)
        plt.plot(aux_bin_activity_nleo, '--')

        aux_offset = aux_offset - 1


def scoreNLEO(Manual_Annotation, NLEO_Binary_Activity, NLEO_Activity):
    NumSignals, L = Manual_Annotation.shape

    # Number of positive and negative annotations
    P = np.sum(Manual_Annotation)
    N = (NumSignals * L) - P

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for n in range(NumSignals):

        aux_manual_annotation = Manual_Annotation[n, :]
        aux_nleo_binary_activity = NLEO_Binary_Activity[n, :]

        y_actual = aux_manual_annotation
        y_hat = aux_nleo_binary_activity

        for i in range(len(y_hat)):
            if y_actual[i] == y_hat[i] == 1:
                TP += 1
            if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
                FP += 1
            if y_actual[i] == y_hat[i] == 0:
                TN += 1
            if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
                FN += 1

    # Precision
    if (TP + FP) == 0:
        Precision = 1
    else:
        Precision = TP / (TP + FP)
        # Recall, Sensitivity
    Recall = TP / P
    # FPR
    Missing = FP / N
    # Mean Error
    Mean_Abs_Error = np.mean(np.abs(Manual_Annotation - NLEO_Binary_Activity))
    # RMSE
    RMSE = np.sqrt(np.mean(np.power(Manual_Annotation - NLEO_Binary_Activity, 2)))
    # F1score
    F1score = 2 * (Precision * Recall) / (Precision + Recall)

    return Precision, Recall, Missing, Mean_Abs_Error, RMSE, F1score


###############################################################################

def testNLEOWindows(Bipolar_signal, NLEO_threshold, window_len, sigma_value, Min_NLEO=0.0001):
    L = len(Bipolar_signal)

    aux_nleo = np.zeros((L,))
    aux_binary_activity = np.zeros((L,))
    #    NLEO_Binary_Activity = np.zeros((L,))
    #    NLEO_Activity = 0

    b_r = fir_d.fir_remez_lpf(125, 150, 0.5, 32, 1000)

    aux_bipolar = Bipolar_signal

    aux_filtered_bipolar = np.convolve(aux_bipolar, b_r, mode='same')

    for i in range(L):
        if (i == 0):
            aux_prev = aux_filtered_bipolar[i]
        else:
            aux_prev = aux_filtered_bipolar[i - 1]

        if i == L - 1:
            aux_next = aux_filtered_bipolar[i]
        else:
            aux_next = aux_filtered_bipolar[i + 1]

        #        a = aux_bipolar[0,i]*aux_bipolar[0,i] - aux_prev*aux_next
        #        print(a.shape)
        aux_nleo[i] = aux_filtered_bipolar[i] * aux_filtered_bipolar[i] - aux_prev * aux_next

    #    print(aux_nleo.shape)
    windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    f1 = plt.figure(1)
    f2 = plt.figure(2)
    #    aux_offset_increment = 1
    aux_offset = -2

    #    plt.subplot(122)
    plt.figure(1)
    aux_label = 'EGM'
    plt.plot(aux_bipolar, label=aux_label)
    aux_label = 'NLEO'
    plt.plot(aux_nleo, label=aux_label)

    plt.plot(aux_nleo + aux_offset)

    for window_type in windows[1:]:
        #    window_type = 'blackman'
        aux_window = eval('np.' + window_type + '(window_len)')

        plt.figure(2)
        plt.plot(aux_window, label=window_type)

        aux_smooth_nleo = np.convolve(aux_nleo, aux_window / aux_window.sum(), mode='same')
        #        aux_smooth_nleo = aux_smooth_nleo/(np.max(aux_smooth_nleo))
        aux_low_nleo_indices = np.where(aux_smooth_nleo < Min_NLEO)
        aux_smooth_nleo[aux_low_nleo_indices] = 0

        ## 4. Adaptive Thresholding
        #        aux_nleo_max = np.max(aux_smooth_nleo)
        aux_nleo_max = np.std(aux_smooth_nleo)
        aux_threshold = NLEO_threshold * aux_nleo_max

        aux_indices = np.where(aux_smooth_nleo > aux_threshold)
        aux_binary_activity[aux_indices] = 1
        #        NLEO_Binary_Activity[:,] = aux_binary_activity

        aux_activity = len(aux_indices[0]) / L

        #        plt.plot(aux_bipolar+aux_offset)
        plt.figure(1)
        # NLEO + LPF
        aux_label = 'NLEO + LPF -' + window_type + ' (' + str(window_len) + ') - AR = ' + str(aux_activity)
        plt.plot(aux_smooth_nleo + aux_offset, label=aux_label)
        aux_label = window_type + ' Thresholded (' + str(window_len) + ')'
        plt.plot(aux_binary_activity + aux_offset, label=aux_label)
    #        aux_offset = aux_offset - aux_offset_increment

    plt.figure(1)
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.legend()
    plt.show()

    f3 = plt.figure(3)
    aux_label = 'EGM'
    plt.plot(aux_bipolar, label=aux_label)
    aux_label = 'NLEO'
    plt.plot(aux_nleo, label=aux_label)

    aux_binary_activity2 = np.zeros((L,))

    # Gaussian 1D kernel
    aux_gaussian_window = windows.gaussian(window_len, std=sigma_value)
    aux_smooth_nleo2 = np.convolve(aux_nleo, aux_gaussian_window, mode='same')

    aux_nleo_max2 = np.std(aux_nleo)
    #    aux_nleo_max2 = np.std(aux_smooth_nleo2)
    aux_threshold2 = NLEO_threshold * aux_nleo_max2
    aux_indices2 = np.where(aux_smooth_nleo2 > aux_threshold2)
    aux_binary_activity2[aux_indices2] = 1
    aux_activity2 = len(aux_indices2[0]) / L

    # NLEO + LPF
    aux_label = 'NLEO + LPF -' + 'Gaussian' + ' ( \sigma=' + str(sigma_value) + ') - AR = ' + str(aux_activity2)
    plt.plot(aux_smooth_nleo2 + aux_offset, label=aux_label)
    aux_label = 'Gaussian' + ' Thresholded ( \sigma=' + str(sigma_value) + ')'
    plt.plot(aux_binary_activity2 + aux_offset, label=aux_label)
    plt.legend()
    plt.show()

    plt.figure(4)
    plt.stem(aux_gaussian_window)
    plt.title(r"Gaussian window ($\sigma$=" + str(sigma_value) + ")")
    plt.ylabel("Amplitude")
    plt.xlabel("Sample")

    plt.figure(5)
    fs = 1000
    A = fft(aux_gaussian_window, 2048) / (len(aux_gaussian_window) / 2.0)
    freq = np.linspace(-0.5 * fs, 0.5 * fs, len(A))
    response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    plt.plot(freq, response)
    plt.axis([0, 0.5 * fs, -120, 0])
    plt.title(r"Frequency response of the Gaussian window ($\sigma$=" + str(sigma_value) + ")")
    plt.ylabel("Normalized magnitude [dB]")
    plt.xlabel("Frequency [Hz]")


###############################################################################
# NLEO STEPS
###############################################################################
# 1 - calculateNLEORaw
# 2 - calculateNLEOLPF
# 3 - calculateNLEOBinaryActivityThreshold
# 4 - calculateNLEOPostProcessingMerge
# 5 - calculateNLEOPostProcessingDiscard

def calculateNLEORaw(Bipolar_signals, Verbose=False, fs = 500):
    # [NLEO, NLEO_Activity] = calculateNLEO(Bipolar_signals)

    N, L = Bipolar_signals.shape
    NLEO = np.zeros((N, L))

    ## 1. BPF to remove the low freq components is not necessary if using Carto signals
    b_r = fir_d.fir_remez_lpf(125, 150, 0.5, 32, fs)

    ## 2. NLEO
    for n in tqdm(range(N), desc="  NLEO Raw", disable=True):
        aux_bipolar = Bipolar_signals[n, :]

        aux_filtered_bipolar = np.convolve(aux_bipolar, b_r, mode='same')

        aux_nleo = np.zeros((L,))

        for i in range(L):
            if (i == 0):
                aux_prev = aux_filtered_bipolar[i]
            else:
                aux_prev = aux_filtered_bipolar[i - 1]

            if i == L - 1:
                aux_next = aux_filtered_bipolar[i]
            else:
                aux_next = aux_filtered_bipolar[i + 1]

            aux_nleo[i] = aux_filtered_bipolar[i] * aux_filtered_bipolar[i] - aux_prev * aux_next

        NLEO[n, :] = aux_nleo

    return NLEO


###############################################################################
def calculateNLEOLPF(NLEO, window_len=50, sigma_value=7, Min_NLEO=0.0001, Verbose=False):
    N, L = NLEO.shape
    NLEO_LPF = np.zeros((N, L))

    # Gaussian filter
    aux_gaussian_window = windows.gaussian(window_len, std=sigma_value)

    ## 2. NLEO
    for n in range(N):
        aux_nleo = NLEO[n, :]
        aux_nleo = np.reshape(aux_nleo, (L,))

        # Gaussian 1D kernel
        aux_smooth_nleo = np.convolve(aux_nleo, aux_gaussian_window, mode='same')

        aux_low_nleo_indices = np.where(aux_smooth_nleo < Min_NLEO)
        aux_smooth_nleo[aux_low_nleo_indices] = 0

        NLEO_LPF[n, :] = aux_smooth_nleo

    return NLEO_LPF


###############################################################################

def calculateNLEOBinaryActivityThreshold(NLEO_LPF, NLEO_threshold=0.1, Verbose=False):
    N, L = NLEO_LPF.shape
    NLEO_Binary_Activity_Threshold = np.zeros((N, L))

    for n in range(N):
        aux_smooth_nleo = NLEO_LPF[n, :]
        aux_smooth_nleo = np.reshape(aux_smooth_nleo, (L,))
        aux_binary_activity = np.zeros((L,))

        ## 4. Adaptive Thresholding wrt STD(NLEO)
        aux_nleo_std = np.std(aux_smooth_nleo)
        aux_threshold = NLEO_threshold * aux_nleo_std

        aux_indices = np.where(aux_smooth_nleo > aux_threshold)
        aux_binary_activity[aux_indices] = 1

        NLEO_Binary_Activity_Threshold[n, :] = aux_binary_activity

    return NLEO_Binary_Activity_Threshold


###############################################################################

# calculateNLEOPostProcessingMerge
def calculateNLEOPostProcessingMerge(NLEO_Binary_Activity_Threshold, Merge_distance=42, Verbose=False):
    N, L = NLEO_Binary_Activity_Threshold.shape
    NLEO_Binary_Activity_Merged = np.zeros((N, L))

    for n in range(N):
        aux_binary_activity = NLEO_Binary_Activity_Threshold[n, :]

        ## 5. Postprocessing
        #        [Activity_merged, Binary_Activity_merged] = mergeBinaryActivity(Binary_Activity, Merge_distance, Verbose)
        aux_binary_activity = np.reshape(aux_binary_activity, (1, L))
        [aux_activity_merged, aux_binary_activity_merged] = mergeBinaryActivity(aux_binary_activity, Merge_distance,
                                                                                Verbose)

        NLEO_Binary_Activity_Merged[n, :] = aux_binary_activity_merged

    return NLEO_Binary_Activity_Merged


###############################################################################

def calculateNLEOPostProcessingDiscard(NLEO_Binary_Activity_Merged, Outlier_distance=0, Verbose=False):
    N, L = NLEO_Binary_Activity_Merged.shape
    NLEO_Binary_Activity_Merged_Outliers = np.zeros((N, L))
    NLEO_Activity = np.zeros((N, 1))

    for n in range(N):
        aux_binary_activity_merged = NLEO_Binary_Activity_Merged[n, :]
        aux_binary_activity_merged = np.reshape(aux_binary_activity_merged, (1, L))

        # Discard outliers
        [aux_activity_merged_outliers, aux_binary_activity_merged_outliers] = discardOutliersBinaryActivity(
            aux_binary_activity_merged, Outlier_distance, Verbose)

        NLEO_Binary_Activity_Merged_Outliers[n, :] = aux_binary_activity_merged_outliers
        NLEO_Activity[n] = aux_activity_merged_outliers

    return NLEO_Binary_Activity_Merged_Outliers, NLEO_Activity
###############################################################################