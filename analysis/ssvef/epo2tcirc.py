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
    # the following assumes that sssPath is in sss_pca_fif, whose raw_sss.fif
    # contain no event channel. So we get events from corresponding file in sss_fif:
    eventPath = sssPath.replace('sss_pca_fif','sss_fif').replace('_allclean_fil80_','_')
    eventSss =  mne.io.read_raw_fif(eventPath, allow_maxshield='yes')
    events = mne.find_events(eventSss, stim_channel=['STI001'])
    # now read the 'sss_pca_fif' with desired data
    sss = mne.io.read_raw_fif(sssPath, allow_maxshield='yes')
    picks = mne.pick_types(sss.info, meg=True)
    event_id = None 
    tmin, tmax = -0., 20.
    tRejCrit = dict(grad=8000e-13, mag=8e-12, eog=np.inf, ecg=np.inf)
    epochs = mne.Epochs(sss, events, event_id, tmin, tmax, picks=picks,
        baseline=None, reject=tRejCrit, preload=True)
    return epochs
    
   

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

dataPath = '/mnt/scratch/prek/'
# start with comprehensive list of "run files"
runPaths = glob(dataPath + '*_camp/fixed_hp/pskt/prek_*/sss_pca_fif/*_raw_sss.fif')
# next, the enclosing "session" directories for each run file:
ssnPaths = sorted( list( set( [ os.path.dirname(p) for p in runPaths ] ) ) )
ssnPaths = ssnPaths[5:9] # quickie test
# for each session, get it's list of runPaths
runPaths = [ np.array( runPaths )[bi] for bi in # the following list of "b(oolean) i(ndices)":
            [ [ r.find(s)>-1 for r in runPaths ] for s in ssnPaths ] ]
# create congruent list of corrensponding epochs
epos = [ [ Sss2Epo(p) for p in rps ] for rps in runPaths ]

# now we must somehow protect against error thrown by concatenate_epochs
# when a run has no epochs; also tcirc needs at least two epochs

epos = [ mne.concatenate_epochs(e) for e in epos ] # this throws error when epo[][].get_data().shape[0]==0

#info = mne.io.read_info(tPaths[0])
#info['sfreq']=20.0 # =0.5
#tcircave = mne.EvokedArray( tcircavg, info )
##tcircave.plot_joint(times=[2.0,4.0,6.0,38.0,40.0,42.0],ts_args=dict(xlim=(0,60),ylim=(0,4),scalings=dict(grad=1, mag=1),units=dict(grad='Tcirc', mag='Tcirc')))
#tcircave.plot_joint(times=[1.95,2.00,2.05,5.95,6.00,6.05],ts_args=dict(xlim=(0,8),ylim=(0,4),scalings=dict(grad=1, mag=1),units=dict(grad='Tcirc', mag='Tcirc')))

