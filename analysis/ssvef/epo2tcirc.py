#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:05:05 2019

@author: pettetmw
"""

# epo2xdawn2tcirc.py

import mne
import mnefun
import os
import numpy as np
from mne.externals.h5io import write_hdf5
import matplotlib.pyplot as plt
from glob import glob

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


def Epo2Xdawn(epoPath,xdawnPath=None):
    # Compute and save (into xdawnPath) XDAWN responses to signal and noise,
    # given epoch data in epoPath; returns status string
    
    # Determine destination path
    if xdawnPath == None:
        # try to derive destination file name from source file name
        xdawnPath = epoPath.replace('-epo.fif','_xdawn_ave.fif')
        
    if xdawnPath == epoPath: # e.g., if find/replace fails, or if incorrect args
        # prevent overwriting epoPath
        errMsg = basename(epoPath) + ' --> error: xdawnPath would overwrite epoPath'
        print( errMsg )
        return errMsg
    
    epochs = mne.read_epochs(epoPath)
    epochs.pick_types(meg=True)
    signal_cov = mne.compute_covariance(epochs, method='oas', n_jobs=18)
#    signal_cov = mne.cov.regularize(signal_cov, epochs.info, rank='full')
    
    rank = mne.compute_rank(signal_cov, rank='full', info=epochs.info)
    signal_cov = mne.cov.regularize(signal_cov, epochs.info, rank=rank)
    
    
    xd = mne.preprocessing.Xdawn(n_components=1, signal_cov=signal_cov,
                                 correct_overlap=False, reg='ledoit_wolf')
    xd.fit(epochs)

    # fit() creates decomposition matrix called filters_ (or "unmixing")
    # and the inverse, called "patterns_" for remixing responses after
    # supressing contribution of selected filters_ components

    # apply() method restricts the reponse to those projected on the
    # filters_ components specified in the "include" arg.  The first tNFC
    # components are the "signal" for purposes of SSNR optimization.

    # calc the signal reponses, as Evoked object
    # (by default, include=list(np.arange(0,1)), i.e., includes only one
    # "signal" component)
    signal = xd.apply(epochs)['tone'].average() # 'tone' is for babies;
                                                # use 'Auditory' for bad_000 pilot

    # calc the noise responses, as Evoked object
    noiseinclude = list(np.arange(1, epochs.info['nchan']))  # a range excluding signal "0"
    noise = xd.apply(epochs, include=noiseinclude)['tone'].average()

    ## create arg to force both plots to have same fixed scaling
    #ts_args = dict(ylim=dict(grad=[-100, 100], mag=[-500, 500]))
    #signal.plot_joint(ts_args=ts_args)
    #noise.plot_joint(ts_args=ts_args)

    ## fit() also computes xd.evokeds_ which seems to be the same as
    ## epochs.average(), but it's calculated in a complicated way that
    ## compensates for overlap (when present).
    ## Keep this handy to compare with xdawn results.
    #xd.evokeds_['Auditory'].average().plot_joint(ts_args=ts_args)
    #epochs.average().plot_joint(ts_args=ts_args)

    # save signal and noise
    # first, replace "Auditory" tag with "signal" and "noise"
    signal.comment = 'signal'
    noise.comment = 'noise'
    
    try:
        mne.write_evokeds( xdawnPath, [ signal, noise ] )
    except:
        errMsg = basename(epoPath) + ' --> error writing ' + xdawnPath
        print( errMsg )
        return errMsg
    
    # Everything worked, so return status string
    return basename(epoPath) + ' --> ' + basename(xdawnPath)

def Xdawn2Tcirc(xdawnPath,tmin=None,tmax=None,fundfreqhz=None):
    # create tcirc stats from xdawn 'signal' and 'noise' that have been
    # saved into xdawnPath by Epo2Xdawn
    
    signal = mne.read_evokeds(xdawnPath,allow_maxshield=True)[0] # "[0]" is 'signal'
    signalTcirc=Tcirc(signal.data,signal.info['sfreq'],tmin=0.5,tmax=1.0,fundfreqhz=20.) # signal t-circ stats
    noise = mne.read_evokeds(xdawnPath,allow_maxshield=True)[1] # "[1]" is 'noise'
    noiseTcirc=Tcirc(noise.data,noise.info['sfreq'],tmin=0.5,tmax=1.0,fundfreqhz=20.) # noise t-circ stats
    
    signalFtzs=None # fisher transformed z-score stats,
                    # for estimating within-subject longitudinal significance
    noiseFtzs=None
    sfreqhz=None # sampling frequency in Hz (from Evoked.info['sfreq']?)
    
    # save the results
    tcircPath = xdawnPath.replace('_xdawn_ave.fif','_tcirc.h5')
    # if find/replace fails, prevent overwriting the input file
    # this needs better solution
    assert xdawnPath == tcircPath
    write_hdf5(tcircPath,
               dict(signaltcirc=signalTcirc,
                    signalftzs=signalFtzs,
                    noisetcirc=noiseTcirc,
                    noiseftzs=noiseFtzs,
                    sfreqhz=sfreqhz),
               title='tcirc', overwrite=True)

def Tcirc(epoOrAve,tmin=None,tmax=None,fundfreqhz=None):
    # create tcirc stats from epoched or evoked file epoOrAve
    # note that np.fft.fft "axis" parameter defaults to -1
    
    # # untested implicit logic for special parameter values
    # if tmin==None, tmin= 0
    # if tmax==None, tmax= end of data along time dim
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
    imx = int( sfreq * tmax )
    
    
    # if evoked, reshape it into fakey epochs, based on fundfreqhz
    if type(epoOrAve) == mne.evoked.Evoked:
        tY = epoOrAve.data[ :, imn:imx ] # 
        tNCh = tY.shape[0]
        tNTrl = int( ( tmax - tmin ) * fundfreqhz ) # the number of "trials"
        tY = np.reshape( tY, (tNCh, tNTrl, -1 ) )
        tY = np.transpose(tY,(1,0,2)) # Trials-by-Chan-by-Freq
#        plt.plot(np.mean(tY,axis=0).T,'r-')
    else:
        tY = epoOrAve.get_data()[ :, :, imn:imx ] # remove odd sample (make this conditional)
        # more needed here to select time window

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

dataPath = '/mnt/scratch/badbaby/tone/'

#tPaths = sorted(glob(dataPath + 'bad*a/epochs/*epo.fif')) # get the existing a's
tPaths = sorted(glob(dataPath + 'bad*b/epochs/*epo.fif')) # get the existing a's
#tPaths = tPaths[5:8] # quickie test

## for comparison, the adult pilot
#tPaths = sorted(glob(dataPath + 'bad_000/epochs/bad_000_tone_epo.fif')) # bad_000 is Adult pilot (E Larson)

tcircs = [ Tcirc(mne.read_epochs(p),tmin=0.5,tmax=1.0) for p in tPaths ]
tcircavg = np.mean( np.stack( tcircs, 2 ), 2 )
#
info = mne.io.read_info(tPaths[0])
info['sfreq']=0.5;
tcircave = mne.EvokedArray( tcircavg, info )
tcircave.plot_joint(times=[2.0,4.0,6.0,38.0,40.0,42.0],ts_args=dict(xlim=(0,60),ylim=(0,4),scalings=dict(grad=1, mag=1),units=dict(grad='Tcirc', mag='Tcirc')))

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

