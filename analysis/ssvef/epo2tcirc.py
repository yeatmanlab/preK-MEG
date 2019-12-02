#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:05:05 2019

@author: pettetmw
"""

# epo2tcirc.py

import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import mne
import mnefun
from mne.externals.h5io import write_hdf5

def Sss2Epo(sssPath):
    # needed to handle pilot 'bad_000' (larson_eric)
    sss = mne.io.read_raw_fif(sssPath, allow_maxshield='yes')
    events = mne.find_events(sss)
    picks = mne.pick_types(sss.info, meg=True)
    event_id = {'Auditory': 1} # 'Auditory' only for bad_000; babies are 'tone'; cf Epo2Xdawn
    tmin, tmax = -0.2, 1.1 # from process.py
    decim = 3 # from process.py
    epochs = mne.Epochs(sss, events, event_id, tmin, tmax, picks=picks,
        decim=decim, baseline=(None, 0), reject=dict(grad=4000e-13),
        preload=True)
    epoPath = sssPath.replace('sss_fif','epochs').replace('_raw_sss.fif','_epo.fif')
    # if find/replace fails, prevent overwriting the input file
    # this needs better solution
    assert sssPath == epoPath 
    epochs.save(epoPath)
    
    # merge above from badbaby, with following from PreK
    
#def GetSsnData( aPFNmPattern ):
##%%
#    # start with desired duration for each segment of the 20-sec trial
#    tDur = 20
#    fSplitEvent = lambda aEv: aEv + np.arange(0,20,tDur) * int(tSR)
#    fRaw = lambda aFile: mne.io.Raw( aFile, allow_maxshield=True, preload=True )
#    fFindEvents = lambda aRaw: mne.find_events( aRaw, stim_channel=['STI001'])
#    tRawPFNms = sorted( glob.glob( aPFNmPattern ) )
#    tRaws = mne.concatenate_raws( [ fRaw( tF ) for tF in tRawPFNms ] )
#    tSR = tRaws.info['sfreq']
#    tEvs = fFindEvents( tRaws )
##    tEvs = np.array([ [ i, 0, 5 ] for i in np.hstack([ fSplitEvent(E) for E in tEvs[:,0] ]) ])
#    
#    #lowpass = 40.
#    #highpass = 0.5
#    #tRaws.filter(highpass, lowpass, fir_design='firwin')
#    
#    
#    #%%
#    # default: dict(mag=1e-12)
##    tRejCrit = dict(grad=3000e-13, mag=3e-12, eog=np.inf, ecg=np.inf)
##    tRejCrit = dict(grad=4000e-13, mag=4e-12, eog=np.inf, ecg=np.inf)
#    tRejCrit = dict(grad=8000e-13, mag=8e-12, eog=np.inf, ecg=np.inf)
##    tRejCrit = dict(grad=np.inf, mag=np.inf, eog = np.inf, ecg = np.inf)
##    
#    # ECG and EOG projections
#    tECG = mne.preprocessing.compute_proj_ecg( tRaws, n_grad=1, n_mag=1, n_eeg=0, average=False, reject=tRejCrit)[0]
#    tEOG = mne.preprocessing.compute_proj_eog(tRaws, n_grad=1, n_mag=1, n_eeg=0, average=False, reject=tRejCrit)[0]
#    tRaws.info['projs'] += tECG  + tEOG
#    
#    video_delay = .066
#    
#    # create epochs
#    tSsn = mne.Epochs(tRaws, tEvs, event_id=None, tmin=0., tmax=tDur, proj=True, baseline=None, reject=tRejCrit)
##    tSsn.plot( scalings=dict(grad=4000e-13, mag=4e-12) );
##    tSsn.plot();
    

def Tcirc(epoOrAve,tmin=None,tmax=None,fundfreqhz=None):
    # create tcirc stats from epoched or evoked file epoOrAve
    # note that np.fft.fft "axis" parameter defaults to -1
    
    # # untested implicit logic for special parameter values
    if tmin==None: tmin= 0  # careful, tmin == 0 implies beginning of baseline?
    if tmax==None: tmax= -1
    # if fundfreqhz == None, fundfreqhz = 1 / (tmax-tmin), an appropriate
    #   default if tcirc is to be estimated from epoch data
    #   For evokeds, use fundfreqhz = N / (tmax-tmin) to divide into N epochs
    
    # e.g., for ASSR: ... = Tcirc(...,tmin=0.5,tmax=1.0,fundfreqhz=20.),
    # will divide the 0.5 second epoch into ten epochs each 1/20.==0.05 sec duration
    
    # Be careful about trailing samples when converting from time to array
    # index values

    #plt.plot(np.mean(tY,axis=-1),'r-')
    
    sfreq = epoOrAve.info['sfreq'] # to compute location of tmin,tmax
    imn = int( sfreq * tmin )
    imx = -1
      
    # if evoked, reshape it into fakey epochs, based on fundfreqhz
    if type(epoOrAve) == mne.evoked.Evoked:
        tY = epoOrAve.data
        if tmax==-1:
            tmax = tY.shape[1] / sfreq
        else:
            imx = int( sfreq * tmax )
        tY = tY[ :, imn:imx ] # 
        if fundfreqhz == None:
            fundfreqhz = 1 / (tmax-tmin)
        tNCh = tY.shape[0]
        tNTrl = int( ( tmax - tmin ) * fundfreqhz ) # the number of "trials"
        tY = np.reshape( tY, (tNCh, tNTrl, -1 ) )
        tY = np.transpose(tY,(1,0,2)) # Trials-by-Chan-by-Freq
#        plt.plot(np.mean(tY,axis=0).T,'r-')
    else:
#        tY = epoOrAve.get_data()[ :, :, imn:imx ] # remove odd sample (make this conditional)
        # more needed here to select time window
        tY = epoOrAve.get_data()
        if tmax==-1:
            tmax = tY.shape[1] / sfreq
        else:
            imx = int( sfreq * tmax )
        tY = tY[ :, :, imn:imx ]

    tNTrl, tNCh, tNS = tY.shape # number of trials, channels, and time samples
    
    tMYFFT = np.fft.fft( np.mean( tY, axis=0 ) ) / tNS # FFT of mean over trials of tY, Chan-by-Freq
    tYFFT = np.fft.fft( tY ) / tNS # FFT of tY , Trials-by-Chan-by-Freq
    
    # compute the mean of the variances along real and imaginary axis
    tYFFTV = np.mean( np.stack( ( np.var( np.real(tYFFT), 0 ), np.var( np.imag(tYFFT), 0 ) ) ), 0 )
    #tYFFTV = np.var( abs(tYFFT), 0 )
    numerator = abs(tMYFFT);
    denominator = np.sqrt( tYFFTV / ( tNTrl - 1 ) )
    #tcirc = abs(tMYFFT) / np.sqrt( tYFFTV / ( tNTrl - 1 ) )
    tcirc = numerator / denominator
    
    return tcirc 

#dataPath = '/mnt/scratch/badbaby/tone/'
##tPaths = sorted(glob(dataPath + 'bad*a/epochs/*epo.fif')) # get the existing a's
#tPaths = sorted(glob(dataPath + 'bad*b/epochs/*epo.fif')) # get the existing a's
##tPaths = tPaths[5:8] # quickie test

# e.g.: '/mnt/scratch/prek/pre_camp/twa_hp/erp/prek_1112/epochs/All_80-sss_prek_1112-epo.fif'
dataPath = '/mnt/scratch/prek/pre_camp/twa_hp/erp/'
tPaths = sorted(glob(dataPath + 'prek_*/epochs/*epo.fif')) # get the existing a's
tPaths = tPaths[5:8] # quickie test


## for comparison, the adult pilot
#tPaths = sorted(glob(dataPath + 'bad_000/epochs/bad_000_tone_epo.fif')) # bad_000 is Adult pilot (E Larson)

tcircs = [ Tcirc(mne.read_epochs(p)) for p in tPaths ]
tcircavg = np.mean( np.stack( tcircs, 2 ), 2 )
#
info = mne.io.read_info(tPaths[0])
info['sfreq']=20.0 # =0.5
tcircave = mne.EvokedArray( tcircavg, info )
#tcircave.plot_joint(times=[2.0,4.0,6.0,38.0,40.0,42.0],ts_args=dict(xlim=(0,60),ylim=(0,4),scalings=dict(grad=1, mag=1),units=dict(grad='Tcirc', mag='Tcirc')))
tcircave.plot_joint(times=[1.95,2.00,2.05,5.95,6.00,6.05],ts_args=dict(xlim=(0,8),ylim=(0,4),scalings=dict(grad=1, mag=1),units=dict(grad='Tcirc', mag='Tcirc')))

## for evokeds, you must supply fundfreqhz
#xdawnPath = 'bad_000_tone_xdawn_ave.fif'
#signal = mne.read_evokeds(xdawnPath,allow_maxshield=True,condition='signal') # or 'noise'
#signalTcirc=Tcirc(signal,tmin=0.5,tmax=1.0,fundfreqhz=20.)

## for epoched, fundfreqhz argument ignored, implicitly 1 /(tmax-tmin) Hz
#epoPath = 'bad_000_tone_epo.fif'
#signal = mne.read_epochs(epoPath)
#signalTcirc=Tcirc(signal,tmin=0.5,tmax=1.0)
#
##info = signal.info;
#info = mne.io.read_info(tPaths[0])
#info['sfreq']=0.5;
#tYFFTT = mne.EvokedArray( signalTcirc, info )
#tYFFTT = mne.EvokedArray( signalTcirc, info )
#
#tYFFTT.plot_joint(times=[38.0,40.0,42.0],ts_args=dict(xlim=(0,60),scalings=dict(grad=1, mag=1),units=dict(grad='Tcirc', mag='Tcirc')))


## this pattern can be used for plotting
## borrowed (from https://github.com/ktavabi/badbaby/blob/master/badbaby/Notebooks/SIMMS_tone.ipynb)
## yet to be tested...
#       evoked = read_evokeds(evoked_file,
#        ...
#        evoked.data, evoked.times = calc_plv(good_epochs,
#                                             freq_picks=(39, 40, 41))
#        evokeds.append(evoked)
#    grndavr_fft = grand_average(evokeds)
#    print(name)    
#    fig = grndavr_fft.plot_topomap(times=evoked.times, vmin=0, vmax=1, 
#                                   scale=4, scale_time=1, unit='PLV',
#                                   cmap = 'seismic', res=128, size=2,
#                                   time_format='%d Hz')

