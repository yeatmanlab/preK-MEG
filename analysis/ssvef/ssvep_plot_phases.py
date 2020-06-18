#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Load SSVEP epochs and plot PSDs, phases, etc.
"""

import os
import numpy as np
from scipy.ndimage import convolve1d
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import mne
from analysis.aux_functions import load_paths, load_params

# config paths
data_root, subjects_dir, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'epochs')
fig_dir = os.path.join(results_dir, 'pskt', 'fig', 'phase')
os.makedirs(fig_dir, exist_ok=True)

# load params
_, _, subjects = load_params()

# config other
timepoints = ('pre', 'post')
subdivide_epochs = 5
subdiv = f'-{subdivide_epochs}_sec' if subdivide_epochs else ''


# loop over subjects
for subj in subjects:
    # loop over timepoints
    for timepoint in timepoints:
        # load epochs
        stub = f'{subj}-{timepoint}_camp-pskt{subdiv}'
        epochs = mne.read_epochs(os.path.join(in_dir, f'{stub}-epo.fif'),
                                 proj=True)
        # TODO: separately for PS and KT trials
        evoked = epochs.average()
        spacing = 1. / evoked.info['sfreq']
        # FFT
        spectrum = rfft(evoked.data, workers=-2)
        freqs = rfftfreq(evoked.times.size, spacing)
        magnitudes = np.abs(spectrum)
        phases = np.angle(spectrum)
        # compute the sensor-space PSD
        sensor_psd = np.abs(spectrum)
        # divide each bin by its two neighbors on each side
        weights = [0.25, 0.25, 0, 0.25, 0.25]
        snr_psd = sensor_psd / convolve1d(sensor_psd, mode='constant',
                                          weights=weights)
        # prep figure
        these_freqs = (2, 3, 4, 5, 6, 11, 12)
        n_freqs = len(these_freqs)
        n_columns = 5
        fig = plt.figure(figsize=(4 * n_columns, 4 * n_freqs))
        for ix, freq in enumerate(these_freqs):
            # focus on the freq bin we care about
            bin_idx = np.argmin(np.abs(freqs - freq))
            these_phases = phases[..., bin_idx]
            these_magnitudes = magnitudes[..., bin_idx]

            # find the sensor that best captures {freq} Hz activity
            best_sensor = np.argmax(snr_psd[..., bin_idx], axis=0)
            best_sensor_name = evoked.ch_names[bin_idx]

            # PSD of best sensor
            ax = plt.subplot(n_freqs, n_columns, n_columns * ix + 1)
            ax.plot(freqs, sensor_psd[best_sensor].T, linewidth=1)
            ax.set(xticks=np.arange(0, 25, 2), xlabel='freq (Hz)',
                   ylabel=f'{freq} Hz\nPSD',
                   title=f'best {freq} Hz power:\n{best_sensor_name}')

            # normalized PSD of best sensor
            ax = plt.subplot(n_freqs, n_columns, n_columns * ix + 2)
            ax.plot(freqs, snr_psd[best_sensor].T, linewidth=1)
            ax.set(xticks=np.arange(0, 25, 2), xlabel='freq (Hz)',
                   ylabel=f'{freq} Hz\n"SNR" (a.u.)')
            title = f'best {freq} Hz SNR: {best_sensor_name}'
            if not ix:
                title = (f'PSD divided by sum of 2 bins on each side\n'
                         f'(channel w/ {title})\n')
            ax.set(title=title)

            # phase vs magnitude (polar)
            ax = plt.subplot(n_freqs, n_columns, n_columns * ix + 3,
                             polar=True)
            ax.plot(these_phases, these_magnitudes,  '.', alpha=0.5)
            ax.set(ylim=(0, magnitudes.max()))
            # goodness of fit
            estimates = spectrum[:, bin_idx]
            estimates = np.array((estimates.real, estimates.imag))
            u, s, v = np.linalg.svd(estimates, full_matrices=False)
            gof = 100 * s[0] / s.sum()
            # plot angle of first eigen & GOF
            u = u[:, 0] * s[0]
            u = u[0] + 1j * u[1]
            ax.plot([0, np.angle(u)], [0, np.abs(u)], color='k', zorder=2)
            ax.text(0, 0, f'{round(gof, 1)}%')
            # title
            if not ix:
                ax.set(title='phase vs magnitude\n')

            # phase histogram
            ax = plt.subplot(n_freqs, n_columns, n_columns * ix + 4,
                             polar=True)
            vals, _, _ = ax.hist(these_phases, bins=72)
            ax.set(ylim=(0, vals.max()))
            if not ix:
                ax.set(title='phase histogram (5° bins)\n')

            # weighted phase histogram
            ax = plt.subplot(n_freqs, n_columns, n_columns * ix + 5,
                             polar=True)
            vals, _, _ = ax.hist(these_phases, bins=72,
                                 weights=these_magnitudes)
            ax.set(ylim=(0, vals.max()))
            if not ix:
                ax.set(title='weighted phase histogram (5° bins)\n')

        fig.suptitle(subj, size=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.95, left=0.05, hspace=0.5)
        fname = f'{stub}-phases.pdf'
        fig.savefig(os.path.join(fig_dir, fname))
        plt.close('all')
