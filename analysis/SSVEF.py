#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:34:05 2018

@author: sjjoo
"""

import mne
import os
import matplotlib.pyplot as plt
import numpy as np
import time

#%%

# Load the data, compute PSDs

tDur = 20.
video_delay = .066
lowpass = 40.
highpass = 0.5

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

tRaws = []

for i in [ '1', '2', '3', '4', '5', '6' ]:
#    tPFNm = '/mnt/scratch/r21/pettet_mark/190109/sss_fif/pm_1_2hz_0' + i + '_raw_sss.fif' # Path File Name
#    tPFNm = '/mnt/scratch/r21/pettet_mark/190109/sss_fif/pm_1_5hz_0' + i + '_raw_sss.fif' # Path File Name
#    tPFNm = '/mnt/scratch/r21/pettet_mark/190109/sss_fif/pm_2hz_0' + i + '_raw_sss.fif' # Path File Name
    tPFNm = '/mnt/scratch/r21/pettet_mark/190109/sss_fif/pm_2hz_no_flash_0' + i + '_raw_sss.fif' # Path File Name
    tRaws = tRaws + [ mne.io.Raw( tPFNm, allow_maxshield=True, preload=True ) ]

tRaws = mne.concatenate_raws( tRaws );
#raws.filter(highpass, lowpass, fir_design='firwin')

tEvs = mne.find_events(tRaws, stim_channel=['STI001','STI002','STI003','STI004'])

tRejCrit = dict(grad=4000e-13, mag=4e-12, eog = np.inf, ecg = np.inf)

# ECG and EOG projections
tECG, tmp = mne.preprocessing.compute_proj_ecg( tRaws, n_grad=1, n_mag=1, n_eeg=0, average=False, reject=tRejCrit)
tEOG, tmp = mne.preprocessing.compute_proj_eog(tRaws, n_grad=1, n_mag=1, n_eeg=0, average=False, reject=tRejCrit)
tRaws.info['projs'] += tECG[-4:]  + tEOG[-4:]

# create epochs
tSsn = mne.Epochs(tRaws, tEvs, event_id=None, tmin=0., tmax=tDur, proj=False, baseline=None, reject=None)
tY = tSsn.get_data()
tY = tY[ :, :, :-1 ]
#plt.plot(np.mean(tY[:,201,:],axis=0),'r-')

tNTrl = tY.shape[0];
tNS = tY.shape[-1]; # Number of Samples|Freqs
tXFrq = np.round( np.fft.fftfreq( tNS, 1.0/tSsn.info['sfreq'] ), 2 ) # X Freq values for horizontal axis of plot

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

tFrqP = list(tXFrq).index( 2.0 ) # Frequency Pick, in Hz
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

tChP = mne.pick_types(tSsn.info, meg='grad', eeg=False, eog=False, selection=['MEG0242']) # Channel Pick
plt.figure()
#plt.plot( tXFrq[range(tNS/2)], np.transpose( abs( tMYFFT[tChP,range(tNS/2)] ) ) )
plt.plot( tXFrq[range(tNS/2)], np.transpose( abs( tYFFTT[tChP,range(tNS/2)] ) ) )

#%%
#""" Timing test """
#data_path = '/mnt/scratch/r21/180712/'
#raw_fname = data_path + 'timing_test_01_raw.fif'
#raw = mne.io.Raw(raw_fname,allow_maxshield=True,preload=True)
#
#order = np.arange(raw.info['nchan'])
#order[0] = 306  # We exchange the plotting order of two channels
#order[1] = 308
#order[2] = 320 # to show the trigger channel as the 10th channel.
#raw.plot(n_channels=3, order=order, block=True, scalings='auto')
#
##picks = mne.pick_types(raw.info, meg=False, eeg=False, eog=False, stim=True, misc=True)
#temp_data = raw.get_data(picks=[306,320])
#
#plt.figure(1)
#plt.clf()
#plt.hold(True)
#plt.plot(temp_data[0])
#plt.plot(temp_data[1])
#
#events = mne.find_events(raw)
#trigger = events[:,0]
#trigger = trigger - raw.first_samp
#
#y1 = temp_data[1]
#tt = 100
#test = 1
#onset = []
#onsetdetect = 0
#for i in np.arange(trigger[0],len(y1)):
#    if y1[i] >= 0.1 and test:
#        onset.append(i)
#        onsetdetect = i
#    if i < onsetdetect + tt:
#        test = 0
#    else:
#        test = 1
#onset = np.array(onset[:len(trigger)])
#
#de = onset-trigger

