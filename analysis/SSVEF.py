#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 13:00:00 2019

@author: pettetmw
"""

import mne
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import glob

#%%

# The following is called by list comprehension beginning after this function def
# (see below)

def GetSsnData( aPFNmPattern ):
    tDur = 20
    fSplitEvent = lambda aEv: aEv + np.arange(0,20,tDur) * int(tSR)
    fRaw = lambda aFile: mne.io.Raw( aFile, allow_maxshield=True, preload=True )
    fFindEvents = lambda aRaw: mne.find_events( aRaw, stim_channel=['STI001'])
    tRawPFNms = sorted( glob.glob( aPFNmPattern ) )
    tRaws = mne.concatenate_raws( [ fRaw( tF ) for tF in tRawPFNms ] )
    tSR = tRaws.info['sfreq']
    tEvs = fFindEvents( tRaws )
    tEvs = np.array([ [ i, 0, 5 ] for i in np.hstack([ fSplitEvent(E) for E in tEvs[:,0] ]) ])
    
    #lowpass = 40.
    #highpass = 0.5
    #tRaws.filter(highpass, lowpass, fir_design='firwin')
    
    
    #%%
    tRejCrit = dict(grad=4000e-13, mag=4e-12, eog = np.inf, ecg = np.inf)
    
    # ECG and EOG projections
    tECG = mne.preprocessing.compute_proj_ecg( tRaws, n_grad=1, n_mag=1, n_eeg=0, average=False, reject=tRejCrit)[0]
    tEOG = mne.preprocessing.compute_proj_eog(tRaws, n_grad=1, n_mag=1, n_eeg=0, average=False, reject=tRejCrit)[0]
    tRaws.info['projs'] += tECG  + tEOG
    
    video_delay = .066
    
    # create epochs
    tSsn = mne.Epochs(tRaws, tEvs, event_id=None, tmin=0., tmax=tDur, proj=True, baseline=None, reject=tRejCrit)
    tY = tSsn.get_data()
    tY = tY[ :, :, :-1 ]
    #plt.plot(np.mean(tY[:,201,:],axis=0),'r-')
    
    tNTrl = tY.shape[0];
    tNS = tY.shape[-1]; # Number of Samples|Freqs
    tXFrq = np.round( np.fft.fftfreq( tNS, 1.0/tSR ), 2 ) # X Freq values for horizontal axis of plot
    
    tYFFT = np.fft.fft( tY ) / tNS # FFT of tY, Trials-by-Chan-by-Freq
    tMYFFT = np.fft.fft( np.mean( tY, axis=0 ) ) / tNS # FFT of mean over trials of tY, Chan-by-Freq
    
    tYFFTV = np.mean( np.stack( ( np.var( np.real(tYFFT), 0 ), np.var( np.imag(tYFFT), 0 ) ) ), 0 )
    #tYFFTV = np.var( abs(tYFFT), 0 )
    tYFFTT = abs(tMYFFT) / np.sqrt( tYFFTV / ( tNTrl - 1 ) )
    
    #%%
    
    # Topographic plot of selected Freq, plus two adjacent ones
    
    ch_names = np.array(tRaws.info['ch_names'])
    tChP = mne.pick_types(tSsn.info, meg='grad', eeg=False, eog=False) # Channel Picks
    tChPI = mne.pick_info(tSsn.info, sel=tChP) # Channel Pick Info
    
    tFrqP = list(tXFrq).index( 6.0 ) # Frequency Pick, in Hz
    tFH, tAHs = plt.subplots(1,3)
    #tVMax = 2.0e-13
    #mne.viz.plot_topomap( abs(tMYFFT[tChP,tFrqP-1]), tChPI, names = ch_names[tChP], show_names=True, axes=tAHs[0], vmax=tVMax )
    #mne.viz.plot_topomap( abs(tMYFFT[tChP,tFrqP]), tChPI, names = ch_names[tChP], show_names=True, axes=tAHs[1], vmax=tVMax )
    #mne.viz.plot_topomap( abs(tMYFFT[tChP,tFrqP+1]), tChPI, names = ch_names[tChP], show_names=True, axes=tAHs[2], vmax=tVMax )
    tVMax = 10
    mne.viz.plot_topomap( tYFFTT[tChP,tFrqP-1], tChPI, names = ch_names[tChP], show_names=True, axes=tAHs[0], vmax=tVMax )
    mne.viz.plot_topomap( tYFFTT[tChP,tFrqP], tChPI, names = ch_names[tChP], show_names=True, axes=tAHs[1], vmax=tVMax )
    mne.viz.plot_topomap( tYFFTT[tChP,tFrqP+1], tChPI, names = ch_names[tChP], show_names=True, axes=tAHs[2], vmax=tVMax )
    
    #%%
    
    # Amplitude histogram from seleted channel.
    
    #tChP = mne.pick_types(tSsn.info, meg='grad', eeg=False, eog=False, selection=['MEG0732']) # Channel Pick
    tChP = mne.pick_types(tSsn.info, meg='grad', eeg=False, eog=False, selection=['MEG0123']) # Channel Pick
    plt.figure()
    #plt.plot( tXFrq[range(tNS/2)], np.transpose( abs( tMYFFT[tChP,range(tNS/2)] ) ) )
    plt.plot( tXFrq[range(tNS/2)], np.transpose( tYFFTT[tChP,range(tNS/2)] ) )
    return tYFFTT


#%%

# Now we can execute the function for this list of args...:
tPFNmPatterns = [
        '/mnt/scratch/r21/pettet_mark/190109/sss_fif/pm_1_2hz_0?_raw_sss.fif',
        '/mnt/scratch/r21/pettet_mark/190109/sss_fif/pm_1_5hz_0?_raw_sss.fif',
        '/mnt/scratch/r21/pettet_mark/190109/sss_fif/pm_2hz_0?_raw_sss.fif',
        '/mnt/scratch/r21/pettet_mark/190109/sss_fif/pm_2hz_no_flash_0?_raw_sss.fif'
]
# using the following list comprehension:
tResults = [ GetSsnData( fp ) for fp in tPFNmPatterns ]

# some additional comments from sjjoo's ssvep.py with file locations

#data_path = '/mnt/scratch/r21/ek_short'
#raw_fname1 = data_path + '/sss_fif/ek_short_1_raw_sss.fif'
#raw_fname2 = data_path + '/sss_fif/ek_short_2_raw_sss.fif'
#raw_fname3 = data_path + '/sss_fif/ek_short_3_raw_sss.fif'
#raw_fname4 = data_path + '/sss_fif/ek_short_4_raw_sss.fif'
#
## long...
#data_path = '/mnt/scratch/r21/ek_long'
#raw_fname1 = data_path + '/sss_fif/ek_long_1_raw_sss.fif'
#raw_fname2 = data_path + '/sss_fif/ek_long_2_raw_sss.fif'

#tRaws = []
#
#for i in [ '1', '2', '3', '4', '5', '6' ]:
##    tPFNm = '/mnt/scratch/r21/pettet_mark/190109/sss_fif/pm_1_2hz_0' + i + '_raw_sss.fif' # Path File Name
##    tPFNm = '/mnt/scratch/r21/pettet_mark/190109/sss_fif/pm_1_5hz_0' + i + '_raw_sss.fif' # Path File Name
##    tPFNm = '/mnt/scratch/r21/pettet_mark/190109/sss_fif/pm_2hz_0' + i + '_raw_sss.fif' # Path File Name
#    tPFNm = '/mnt/scratch/r21/pettet_mark/190109/sss_fif/pm_2hz_no_flash_0' + i + '_raw_sss.fif' # Path File Name
#    tRaws = tRaws + [ mne.io.Raw( tPFNm, allow_maxshield=True, preload=True ) ]
#
