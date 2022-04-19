# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 20:30:10 2022

@author: CITI
"""


#%%
import os
import numpy as np

from madmom.features.downbeats import RNNDownBeatProcessor as mmRNN
from pathlib import Path
import modules as plp_mod
import glob

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
f_measure_threshold = 0.07


def qual_plots(songinfo, start_frame=1000, dur=10, fps = 100,
               legend_fontsize=16, fontsize=18, figsize = (14, 3), 
               ):
    """ generate qualitative results using songinfo """
    colors = list(mcolors.TABLEAU_COLORS.keys())
    end_frame = start_frame + dur*fps
    plt.figure(figsize = figsize)
    plt.plot(songinfo.acti_max, color = 'red')
    plt.vlines(songinfo.est_beats_dict['HMM']*100, 1, 0.75, label = 'HMM', 
               color = colors[0], linewidth = 5, alpha = 0.7)
    plt.vlines(songinfo.est_beats_dict['HMMT0']*100, 0.75, 0.5, label = 'HMMT0', 
               color = colors[1], linewidth = 5, alpha = 0.7)
    plt.vlines(songinfo.est_beats_dict['PLPDP']*100, 0.25, 0.5, label = 'PLPDP', 
               color = colors[2], linewidth = 5, alpha = 0.7)
    plt.vlines(songinfo.ref_beats[:,0]*100, 0., 1, label = 'Reference Beats', linestyle = 'dashed')
    plt.xlabel('Time Frame (FPS =100)', fontsize = fontsize)
    plt.ylabel('Amplitude', fontsize = fontsize)
    plt.xticks( fontsize = fontsize)
    plt.yticks( fontsize = fontsize)
    plt.legend(bbox_to_anchor=(0.5, 1.2), loc = 'upper center', 
               fontsize = legend_fontsize, frameon = False, ncol=4)
    plt.xlim([start_frame, end_frame])
    return plt

class song_info():
    def __init__(self, songpath, post_types, ref_folder, 
                 min_bpm = 30, max_bpm = 300, fps = 100, dp_factor = 5):
        self.songpath = songpath
        self.ref_folder = ref_folder
        self.annpath = os.path.join(self.ref_folder, 
                                    os.path.basename(self.songpath).replace('.wav', '.beats'))
        self.acti_max = self.get_acti_max()
        self.post_types = post_types
        self.mean_temop = plp_mod.getSongMeanTempo(self.annpath, 
                                           mean_type = 'ori_ibi', 
                                           smooth_winlen = None)
        self.ppt_params = {'min_bpm':min_bpm, 'max_bpm':max_bpm, 
                           'fps': fps, 'dp_factor': dp_factor}
        self.est_beats_dict = self.get_est_beats()
        
        self.ref_beats = self.get_ref_beats()
        

    def get_ref_beats(self):
        print("calculating ref beats...")
        ref_beats = np.loadtxt(self.annpath)
        return ref_beats
    
    def get_acti_max(self):
        print("calculating madmom activation...")
        acti_max = mmRNN()(self.songpath).max(axis=1)
        return acti_max
    def get_est_beats(self):
        print("calculating estimated beats...")
        est_beats_dict = {}
        for post_type in self.post_types:
            beat_est = plp_mod.acti2Est(self.acti_max, post_type,  
                              dp_meantempo = self.mean_temop, **self.ppt_params )
            est_beats_dict[post_type] = beat_est
        return est_beats_dict

def main():
    ### paths for test songs
    songs = glob.glob(os.path.join('./', 'test_recordings', '*.wav'))
    ### types of Post-processing trackers to use
    post_types = ['SPPK', 'DP', 'PLPDP-sk', 'PLPDP', 'HMM', 'HMMT0']
    ### generate beat estimations and other information for each test song
    songinfo_list = []
    for eachsong in songs[:1]:
        song_info_temp = song_info(eachsong, post_types, './test_recordings')
        songinfo_list.append(song_info_temp)
    
    ### save qualitative plots and beat estimations
    fig_out_dir = os.path.join('./', 'test_recordings', 'qualitative_plots')
    if not os.path.exists(fig_out_dir):
        Path(fig_out_dir).mkdir(parents = True, exist_ok = True)
        
    est_out_dir = os.path.join('./test_recordings', 'estimated_beats', )
    if not os.path.exists(est_out_dir):
        Path(est_out_dir).mkdir(parents =True, exist_ok = True)
    
    for tmp in songinfo_list[:1]:
        songname = os.path.basename(tmp.songpath)
        print("=========Ploting Beat Estimations...=========")
        print("==={}===".format(songname))
        plt = qual_plots(tmp)
        fig_dir = os.path.join(fig_out_dir, songname.replace('.wav', '.png'))
        if not os.path.exists(fig_dir):
            plt.savefig(fig_dir, bbox_inches = 'tight', dpi=600)
        ### save beat estimations
        for post_type in post_types:
            est_dir = os.path.join(est_out_dir, post_type)
            if not os.path.exists(est_dir):
                Path(est_dir).mkdir( parents = True, exist_ok =True)
            est_spath = os.path.join(est_dir, os.path.basename(tmp.annpath))
            if not os.path.exists(est_spath):
                print("saving {}...".format(est_spath))
                np.savetxt(est_spath, tmp.est_beats_dict[post_type], fmt = '%.5f')
#%%
if __name__=="__main__":
    main()


    