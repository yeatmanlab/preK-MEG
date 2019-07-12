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


# The following is called by list comprehension beginning after this function def
# (see below)

def GetSsnData( aPFNmPattern ):
#%%
    # start with desired duration for each segment of the 20-sec trial
    tDur = 20
    fSplitEvent = lambda aEv: aEv + np.arange(0,20,tDur) * int(tSR)
    fRaw = lambda aFile: mne.io.Raw( aFile, allow_maxshield=True, preload=True )
    fFindEvents = lambda aRaw: mne.find_events( aRaw, stim_channel=['STI001'])
    tRawPFNms = sorted( glob.glob( aPFNmPattern ) )
    tRaws = mne.concatenate_raws( [ fRaw( tF ) for tF in tRawPFNms ] )
    tSR = tRaws.info['sfreq']
    tEvs = fFindEvents( tRaws )
#    tEvs = np.array([ [ i, 0, 5 ] for i in np.hstack([ fSplitEvent(E) for E in tEvs[:,0] ]) ])
    
    #lowpass = 40.
    #highpass = 0.5
    #tRaws.filter(highpass, lowpass, fir_design='firwin')
    
    
    #%%
    # default: dict(mag=1e-12)
#    tRejCrit = dict(grad=3000e-13, mag=3e-12, eog=np.inf, ecg=np.inf)
#    tRejCrit = dict(grad=4000e-13, mag=4e-12, eog=np.inf, ecg=np.inf)
    tRejCrit = dict(grad=8000e-13, mag=8e-12, eog=np.inf, ecg=np.inf)
#    tRejCrit = dict(grad=np.inf, mag=np.inf, eog = np.inf, ecg = np.inf)
#    
    # ECG and EOG projections
    tECG = mne.preprocessing.compute_proj_ecg( tRaws, n_grad=1, n_mag=1, n_eeg=0, average=False, reject=tRejCrit)[0]
    tEOG = mne.preprocessing.compute_proj_eog(tRaws, n_grad=1, n_mag=1, n_eeg=0, average=False, reject=tRejCrit)[0]
    tRaws.info['projs'] += tECG  + tEOG
    
    video_delay = .066
    
    # create epochs
    tSsn = mne.Epochs(tRaws, tEvs, event_id=None, tmin=0., tmax=tDur, proj=True, baseline=None, reject=tRejCrit)
#    tSsn.plot( scalings=dict(grad=4000e-13, mag=4e-12) );
#    tSsn.plot();
    
#%%
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

    
    tInf = tSsn.info;
    tInf['sfreq']=20;
    tYFFTT = mne.EvokedArray( tYFFTT, tInf )
    
    #tYFFTT.plot_joint(times=[5.95,6.00,6.05],ts_args=dict(xlim=(5.75,6.25),scalings=dict(grad=1, mag=1),units=dict(grad='Tcirc', mag='Tcirc')))
    
    
    return tYFFTT


#%%
# Now we can execute the function for this list of args...:

tPFNmPatterns = [ x + '/*/*raw_sss.fif' for x in sorted( glob.glob( 'prek_????' ) ) ]


#tPFNmPatterns = [
##        '/mnt/scratch/r21/pettet_mark/190109/sss_fif/pm_1_2hz_0?_raw_sss.fif',
##        '/mnt/scratch/r21/pettet_mark/190109/sss_fif/pm_1_5hz_0?_raw_sss.fif',
##        '/mnt/scratch/r21/pettet_mark/190109/sss_fif/pm_2hz_0?_raw_sss.fif',
##        '/mnt/scratch/r21/pettet_mark/190109/sss_fif/pm_2hz_no_flash_0?_raw_sss.fif'
##        '/mnt/scratch/preK_out/prek_1451_190419/sss_fif/prek_1451_190419_0[1,2,3]_raw_sss.fif'
##        '/mnt/scratch/preK_out/prek_1259_190419/sss_fif/prek_1259_190419_0[1,2,3,4]_raw_sss.fif'
##        '/mnt/scratch/preK_out/jason_yeatman_190514/sss_fif/jason_yeatman_190514_0[1,2,4]_raw_sss.fif'
#        '/mnt/scratch/preK_out/prek_1964/sss_fif/prek_1964_pskt_0[1,2]_pre_raw_sss.fif'
#]

# using the following list comprehension:
tR = [ GetSsnData( fp ) for fp in tPFNmPatterns ]

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


