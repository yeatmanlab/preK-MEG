#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:07:42 2019

@author: mpettet
"""

# following on from xdawn_assr:

tED = epochs;
tSR = tED.info['sfreq'];
tY = tED.get_data();
tY = tY[ :, :, int(0.5*tSR):int(1.0*tSR) ];
tNTrl = tY.shape[0];
tNS = tY.shape[-1]; # Number of Samples|Freqs
tXFrq = np.round( np.fft.fftfreq( tNS, 1.0/tSR ), 2 ) # X Freq values for horizontal axis of plot

tYFFT = np.fft.fft( tY ) / tNS # FFT of tY, Trials-by-Chan-by-Freq
tMYFFT = np.fft.fft( np.mean( tY, axis=0 ) ) / tNS # FFT of mean over trials of tY, Chan-by-Freq

tYFFTV = np.mean( np.stack( ( np.var( np.real(tYFFT), 0 ), np.var( np.imag(tYFFT), 0 ) ) ), 0 )
#tYFFTV = np.var( abs(tYFFT), 0 )
tYFFTT = abs(tMYFFT) / np.sqrt( tYFFTV / ( tNTrl - 1 ) ) # these are the Tcirc scores

ch_names = np.array(tED.info['ch_names'])
tChP = mne.pick_types(tED.info, meg='grad', eeg=False, eog=False) # Channel Picks
tChPI = mne.pick_info(tED.info, sel=tChP) # Channel Pick Info

tFrqP = list(tXFrq).index( 40.0 ) # Frequency Pick, in Hz

tVMax = 10
tFH = plt.figure();
tGS = tFH.add_gridspec(3,5);
fSubPlot = lambda aGS: tFH.add_subplot(aGS);
tAHs = [ fSubPlot(x) for x in [ tGS[0,0], tGS[0,1], tGS[0,2], tGS[0,3], tGS[0,4], tGS[1:,:] ] ];

fTopoPlot = lambda i: mne.viz.plot_topomap( tYFFTT[tChP,tFrqP+i], tChPI, names = ch_names[tChP], show_names=False, axes=tAHs[i+2], vmax=tVMax );
tImHs = [ fTopoPlot(i) for i in [ -2, -1, 0, 1, 2 ] ];
fAxTitle = lambda i: tAHs[i+2].set_title( str( tXFrq[tFrqP+i] ) + " Hz" );
tImHs = [ fAxTitle(i) for i in [ -2, -1, 0, 1, 2 ] ];


tAHs[-1].plot( tChP, tYFFTT[tChP,(tFrqP-2):(tFrqP+3)] );
tAHs[-1].legend( [ str(x)+" Hz" for x in tXFrq[(tFrqP-2):(tFrqP+3)] ] );
tAHs[-1].set_xlabel('Channel ID Number')
tAHs[-1].set_ylabel('Tcirc score')
