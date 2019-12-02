#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 13:00:00 2019

@author: pettetmw
"""

import mne
import os
import numpy as np
import glob

import matplotlib.pyplot as plt
import time


def IfMkDir( aPNm ):
    if not os.path.exists( aPNm ):
        os.mkdir( aPNm )

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
    
    fGrandParentPNm = lambda x: os.path.dirname(os.path.dirname( x ) )
    tSbjDir = fGrandParentPNm( aPFNmPattern )
    tSbjID = os.path.basename( tSbjDir )
    IfMkDir( os.path.join( tSbjDir, 'tcirc_fif' ) )
    tTcircPFNm = os.path.join( tSbjDir, 'tcirc_fif', tSbjID + '_tcirc.fif' )
    
    tYFFTT.save( tTcircPFNm )

    return tYFFTT

#%%
# Now we can execute the function for this list of args...:

# the following is only appropriate for /mnt/scratch/prek/visit1
tPFNmPatterns = [ x + '/*/*raw_sss.fif' for x in sorted( glob.glob( 'prek_????' ) ) ]

# for actual data in /mnt/scratch/prek/p*/fixed_hp/prek_????/sss_pca_fif/*raw_sss.fif


# using the following list comprehension:
tR = [ GetSsnData( fp ) for fp in tPFNmPatterns ]
# to create and save...

#%%
#... the following files
tPFNmPatterns = sorted( glob.glob( 'prek_????/*/*tcirc.fif' ) );

fLoadEvoked = lambda aFNm: mne.Evoked( aFNm, allow_maxshield=True )
# which we can later reload:
tR = [ fLoadEvoked( fp ) for fp in tPFNmPatterns ]
tR = [ r for r in tR if not np.isnan(r.data).all().all() ]
# and compute visualization of grand average
oEAvg = mne.grand_average( tR )
oEAvg.plot_joint(times=[5.95,6.00,6.05],ts_args=dict(xlim=(5.75,6.25),ylim=(0.0,20.0),scalings=dict(grad=1, mag=1),units=dict(grad='Tcirc', mag='Tcirc')))
oEAvg.plot_joint(times=[1.95,2.00,2.05],ts_args=dict(xlim=(1.75,2.25),ylim=(0.0,20.0),scalings=dict(grad=1, mag=1),units=dict(grad='Tcirc', mag='Tcirc')))

#fPNan = lambda x: np.isnan(x.data).sum() / x.data.size
#[ fPNan(r) for r in tR ]
#Out[5]: 
#[0.023627507598784195,
# 0.020588145896656536,
# 0.023625379939209726,
# 0.02373996960486322,
# 0.014621428571428572,









