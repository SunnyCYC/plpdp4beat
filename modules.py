# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:47:52 2022

20230627: add noew plp_int_tl to limit tempo range (theta) when using PLP kernel size of 1 sec

@author: CITI
"""
#%%
import numpy as np
from madmom.features.beats import DBNBeatTrackingProcessor as Bproc 
from scipy.signal import find_peaks
import libfmp.c6 as libfmp
import os 



def compute_plps_norm(X, Fs, L, N, H, Theta, plp_num = 3):
    """Compute windowed sinusoid with optimal phase 
    Note: original source code is from: 
        https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S3_PredominantLocalPulse.html
    And we add a normalization process in this code.
    
    Args:
        X (np.ndarray): Fourier-based (complex-valued) tempogram
        Fs (scalar): Sampling rate
        L (int): Length of novelty curve
        N (int): Window length
        H (int): Hop size
        Theta (np.ndarray): Set of tempi (given in BPM)
        plp_num (int): num of output plp curves (e.g. 2--> output 2 largest plp)

    Returns:
        nov_PLP (np.ndarray): PLP function
    """
    win = np.hanning(N)
    ## normalization 
    win = win/(sum(win)/len(win))
    win = win/(len(win)/H)
    N_left = N // 2
    L_left = N_left
    L_right = N_left
    L_pad = L + L_left + L_right
    
    M = X.shape[1]
    tempogram = np.abs(X)
    
    nov_PLPs = np.zeros(( plp_num, L_pad), dtype = np.float64)
    multi_tempo_list = []
    for n in range(M):
        ## get the predominant tempos
        ### deal with zero input:
        if not tempogram[:, n].sum() ==0:   
            rank = np.argsort(-tempogram[:, n])
            tempos = Theta[rank[0:plp_num]]
            multi_tempo_list.append(tempos)
            omegas = (tempos/60)/Fs
            cs = X[rank[0:plp_num], n]
            phases = - np.angle(cs) / (2 * np.pi)
            
            t_0 = n * H
            t_1 = t_0 + N
            t_kernel = np.arange(t_0, t_1)
            kernels = win*np.cos(2 * np.pi * (t_kernel[np.newaxis, :] * omegas[:, np.newaxis] - phases[:, np.newaxis]))
        else:
            ## if zero input, nov_PLPs will remains zero in this region
            t_0 = n * H
            t_1 = t_0 + N
            t_kernel = np.arange(t_0, t_1)
            kernels = 0*(t_kernel[np.newaxis, :])
            
        nov_PLPs[:, t_kernel] = nov_PLPs[:, t_kernel] + kernels
        
    nov_PLPs = nov_PLPs[:, L_left:L_pad-L_right]
    nov_PLPs[nov_PLPs < 0] = 0
    
    
    return nov_PLPs, np.array(multi_tempo_list)

def plp2bref(plp_cur, PK_Height = 0.1 , prominence = 0.1, distance = 7, 
                  return_peak = False):
    """
    Covert a PLP curve into 2D tempo related condition.

    Parameters
    ----------
    plp_cur : (np.ndarray)
        PLP curve.
    PK_Height : float, optional
        Peak threshold for peak picking function. The default is 0.1.
    prominence : float, optional
        Prominence for peak picking function. The default is 0.1.
    distance : int, optional
        Distance for peak picking function. The default is 7.
    return_peak : TYPE, optional
        Return or not the peaks detected by peak picking. The default is False.

    Returns
    -------
    
    ref_table: (np.ndarray)
        2D array containing time varying IBI and Confidence

    """
    peaks, properties = find_peaks(plp_cur, height = PK_Height, 
                                   prominence = prominence, distance = distance)
    
    ref_table = np.zeros((2, len(plp_cur))) # beatref, confidence
    
    ### start from second peak, calculate beat_ref 
    start_frame = 0
    for p_ind in range(1, len(peaks)):

        ### find right bases
        rbs_pos = properties['right_bases'][p_ind]
        cur_p_pos = peaks[p_ind]
        prev_p_pos = peaks[p_ind -1]
        next_p_pos = peaks[p_ind +1] if (p_ind+1)<len(peaks) else len(plp_cur)-1
        
        ### if right base is too far away, use midpoint between two peaks
        if rbs_pos>next_p_pos:
            rbs_pos = int((cur_p_pos+next_p_pos)/2)
        # print("start_frame:{}, right_base:{}".format(start_frame, rbs_pos))
       
        ## save beat_ref
        beat_ref = cur_p_pos - prev_p_pos
        ref_table[0, start_frame:rbs_pos] = beat_ref
        ## save confidence
        conf = (properties['peak_heights'][p_ind-1:p_ind+1].mean())
        ref_table[1, start_frame:rbs_pos] = conf
        ## shift start_frame
        start_frame = rbs_pos
    ### last few frames
    rbs_pos = len(plp_cur)
    ref_table[0, start_frame:rbs_pos] = beat_ref
    ref_table[1, start_frame:rbs_pos] = conf
    if return_peak:
        return ref_table, peaks
    else:
        return ref_table

def nov2plp(nov, kernel_size = 3, plp_num = 1, 
            Fs_nov = 100, H = 10, Theta =np.arange(30, 301) ):
    """
    Compute PLP curves from input novelty function

    Parameters
    ----------
    nov : (np.ndarray)
        A novelty function such as output of onset detection or network activation.
    kernel_size : int, optional
        Window size of PLP in seconds. The default is 3.
    plp_num : int, optional
        The number of PLP curves to derive. The default is 1.
    Fs_nov : int, optional
        Sampling rate of novelty function in frame-per-second. The default is 100.
    H : int, optional
        Hop size for PLP kernel/window in frames. The default is 10.
    Theta : (np.ndarray), optional
        Tempo range for tempogram (in BPM). The default is np.arange(30, 301).

    Returns
    -------
    plp_curve : (np.ndarray)
        Derived PLP curves in shape (curve id, time instants).

    """
    L = len(nov)
    N = kernel_size*100

    X, T_coef, F_coef_BPM = libfmp.compute_tempogram_fourier(nov, 
                                                                Fs=Fs_nov, 
                                                                N=N, H=H, 
                                                                Theta=Theta) 

    nov_PLPs, f_mtempo_list = compute_plps_norm(X, Fs_nov, L, N, H, Theta, plp_num = plp_num)
    plp_curve = nov_PLPs[0:plp_num, :]
    return plp_curve

def plpdp(novelty, ref_tab, penalty=None, return_all=False):
    """
    PLPDP modified from the following DP code: 
        https://github.com/meinardmueller/libfmp/blob/master/libfmp/c6/c6s3_beat_tracking.py
        https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S3_BeatTracking.html
        
    | Compute beat sequence using dynamic programming [FMP, Section 6.3.2]
    | Note: Concatenation of '0' because of Python indexing conventions

    Notebook: C6/C6S3_BeatTracking.ipynb

    Args:
        novelty (np.ndarray): Novelty function (could be output of DL models)
        ref_tab (np.ndarray): reference table of beatref and confidence (# beatref, confidence)
        penalty (np.ndarray): Penalty function (Default value = None)

        return_all (bool): Return details (Default value = False)

    Returns:
        B (np.ndarray): Optimal beat sequence
        D (np.ndarray): Accumulated score
        P (np.ndarray): Maximization information
    """
    N = len(novelty)
    ### padding to make start index be 1
    novelty = np.concatenate((np.array([0]), novelty))
    ref_tab = np.concatenate((np.zeros((2, 1)), ref_tab), axis =1)

    prev_bref = ref_tab[0, 1] ## ref_tab's bref are in frames not bpm

    if penalty is None:
        penalty = libfmp.compute_penalty(N, prev_bref)
    factor = ref_tab[1, 1]
    penalty = penalty * factor
    
    
    D = np.zeros(N+1)
    P = np.zeros(N+1, dtype=int)
    D[1] = novelty[1]
    P[1] = 0
    # forward calculation
    for n in range(2, N+1):
        m_indices = np.arange(1, n)

        cur_bref = ref_tab[0, n]
        ### calculate penalty if bref diffs in ref_tab
        if cur_bref !=prev_bref:
            penalty = libfmp.compute_penalty(N, cur_bref)
            factor = ref_tab[1, n]
            penalty = penalty * factor  
            prev_bref = cur_bref

        scores = D[m_indices] + penalty[n-m_indices]
        maxium = np.max(scores)
        if maxium <= 0:
            D[n] = novelty[n]
            P[n] = 0
        else:
            D[n] = novelty[n] + maxium
            P[n] = np.argmax(scores) + 1
    # backtracking
    B = np.zeros(N, dtype=int)
    k = 0
    B[k] = np.argmax(D)
    while P[B[k]] != 0:
        k = k+1
        B[k] = P[B[k-1]]
    B = B[0:k+1]
    B = B[::-1]
    B = B - 1
    if return_all:
        return B, D, P
    else:
        return B  

def getSongMeanTempo(beat_file, mean_type = 'ori_ibi', smooth_winlen = None):
    """
    Calculate mean tempo based on mean IBI of reference beats in input beat_file
    
    Parameters
    ----------
    beat_file : str
        path of beat annotation for a specific song.
    mean_type : str, optional
        method name for calculating mean_tempo of the song. The default is 'ori_ibi'.
    smooth_winlen : int, optional
        if requires smoothing when using different mean_type, could be use to decide
        window length. The default is None.

    Returns
    -------
    mean_tempo: float
        Mean tempo of the input beat_file in BPM.


    """
    beats = np.loadtxt(beat_file)
    
    if len(beats.shape)<2:
        beats = beats[:, np.newaxis]
    ### if the annotation contains less than 2 beats:
    if len(beats)<2:
        print("===> Warning: Less than 2 beats within:{}".format(beat_file))
        print("===> return 0 BPM")
        return 0
    else:
        ibis = beats[1:, 0]-beats[0:-1, 0]
        ### mean tempo of each song can be calculated in different ways
        ### "ori_ibi": using raw inter-beat-intervals without any smoothing
        if mean_type =='ori_ibi':
            mean_ibi = ibis.mean() # in sec
            mean_tempo = 60/mean_ibi
        else:
            print("Haven't implement this mean_type:", mean_type)
            mean_tempo =  None
    
        return mean_tempo

def plp_int(nov, fps = 100, theta = np.arange(30, 301)):
    """
    Calculate integrated PLP using kernel sizes of 1, 3, 5 seconds.

    Parameters
    ----------
    nov : (np.ndarray)
        Novelty function such as results of onset detection or network beat activation.
    fps : float, optional
        Frame rate of the input novelty function. The default is 100.
    theta : (np.ndarray), optional
        Tempo range for tempogram. The default is np.arange(30, 301).

    Returns
    -------
    plp_int : (np.ndarray)
        An integrated PLP curve.

    """
    ### to ensure plp_k1 at least one completed sinewave in one window, set the min_bpm = 60 bpm
    if theta[0] <60:    
        theta_k1 = np.arange(60, theta[-1]+1)
    else:
        theta_k1 = theta
    plp_k1 = nov2plp(nov = nov, kernel_size = 1, plp_num = 1, 
                                        Fs_nov = fps, H = 10, Theta = theta_k1)
    plp_k3 = nov2plp(nov = nov, kernel_size = 3, plp_num = 1, 
                                        Fs_nov = fps, H = 10, Theta = theta)
    plp_k5 = nov2plp(nov = nov, kernel_size = 5, plp_num = 1, 
                                        Fs_nov = fps, H = 10, Theta = theta)
    plp_int = (plp_k1*plp_k3*plp_k5).squeeze()

    return plp_int

def plp_int_tl(nov, fps = 100, theta = np.arange(30, 301)):
    """
    Calculate integrated PLP using kernel sizes of 1, 3, 5 seconds.
    0627: tempo range limited version for PLP with kernel size of 1 second

    Parameters
    ----------
    nov : (np.ndarray)
        Novelty function such as results of onset detection or network beat activation.
    fps : float, optional
        Frame rate of the input novelty function. The default is 100.
    theta : (np.ndarray), optional
        Tempo range for tempogram. The default is np.arange(30, 301).

    Returns
    -------
    plp_int : (np.ndarray)
        An integrated PLP curve.

    """
    ### to ensure plp_k1 at least capture 2 peaks in one window, set the min_bpm = 60 bpm
    if theta[0] <60:    
        theta_k1 = np.arange(60, theta[-1]+1)
    else:
        theta_k1 = theta
    plp_k1 = nov2plp(nov = nov, kernel_size = 1, plp_num = 1, 
                                        Fs_nov = fps, H = 10, Theta = theta_k1)
    plp_k3 = nov2plp(nov = nov, kernel_size = 3, plp_num = 1, 
                                        Fs_nov = fps, H = 10, Theta = theta)
    plp_k5 = nov2plp(nov = nov, kernel_size = 5, plp_num = 1, 
                                        Fs_nov = fps, H = 10, Theta = theta)
    plp_int = (plp_k1*plp_k3*plp_k5).squeeze()

    return plp_int

def acti2Est(max_acti, post_type, min_bpm = 30, max_bpm =300, 
                 dp_meantempo = None, fps = 100, dp_factor = 5,  ):
    """
    Get beat estimation based on input max-acti and the assigned post_type (PPT)

    Parameters
    ----------
    max_acti : (np.ndarray)
        Activation function of novelty function. Can be derived via taking maximum
        of each time frame of madmom 2D activation function for beat and downbeat.
    post_type : str
        Post processing tracker's name. 
        Options: ['DP', 'SPPK','PLPDP-dk', 'PLPDP', 'HMM', 'HMMT0']
    min_bpm : int, optional
        Minimum Tempo (BPM) for Post-processors to set search spaces. The default is 30.
    max_bpm : int, optional
        Maximum Tempo (BPM) for Post-processors to set search spaces. The default is 300.
    dp_meantempo : float, optional
        Global tempo as a required condition for 'DP' PPT. The default is None.
    fps : int, optional
        Frame rate of the input novelty function, max_acti. The default is 100.
    dp_factor : float, optional
        A factor for DP to determine the balance between its two assumptions. The default is 5.


    Returns
    -------
    beat_est : (np.ndarray)
        Estimated beats in seconds.

    """
    theta = np.arange(min_bpm, max_bpm+1)


        
    if post_type =='SPPK':
        beats_spppk_tmp, _ = find_peaks(max_acti, height = 0.1, 
                                       distance = 7, 
                                       prominence = 0.1)
        beat_est = beats_spppk_tmp/fps
    elif post_type =='DP':
        
        beat_ref = (60/dp_meantempo)*fps
        beat_est = libfmp.compute_beat_sequence(max_acti, beat_ref = beat_ref, 
                                         factor = dp_factor)/fps
    elif post_type =='PLPDP-sk':
        acti_fplp = nov2plp(nov = max_acti, kernel_size = 3, plp_num = 1, 
                                    Fs_nov = fps, H = 10, Theta = theta)
        ref_tab = plp2bref(acti_fplp.squeeze(), PK_Height = 0.1, prominence = 0.1, 
            distance = 7)
        beat_est = plpdp(max_acti, ref_tab, return_all = False)
        
        beat_est = beat_est/fps   
    elif post_type =='PLPDP': 
        plp_int_curf = plp_int(nov= max_acti, fps = fps, theta = theta)
        ref_tab = plp2bref(plp_int_curf, PK_Height = 0.1, prominence = 0.1, 
            distance = 7)
        beat_est = plpdp(max_acti, ref_tab, return_all = False)
        beat_est = beat_est/fps
    elif post_type =='HMM':
        hmm_proc= Bproc( fps = fps, 
                         max_bpm=max_bpm, min_bpm=min_bpm)

        beat_est = hmm_proc(max_acti)

    elif post_type =='HMMT0':
        hmm_proc= Bproc( transition_lambda=0,
                             fps = fps, 
                         max_bpm=max_bpm, min_bpm=min_bpm)

        beat_est = hmm_proc(max_acti)

    else:
        print("unknown post_type:", post_type)
        

    return beat_est
