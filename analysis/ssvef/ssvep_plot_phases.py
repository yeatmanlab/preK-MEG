#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Load SSVEP epochs and plot PSDs, phases, etc.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from analysis.aux_functions import load_paths, load_params, div_by_adj_bins

# config paths
data_root, subjects_dir, results_dir = load_paths()
fft_dir = os.path.join(results_dir, 'pskt', 'fft-evoked')
fig_dir = os.path.join(results_dir, 'pskt', 'fig', 'phase')
os.makedirs(fig_dir, exist_ok=True)

# load params
*_, subjects, cohort = load_params(experiment='pskt')

# config other
timepoints = ('pre', 'post')


# loop over subjects
for subj in subjects:
    # loop over timepoints
    for timepoint in timepoints:
        for condition in ('ps', 'kt', 'all'):
            stub = f'{subj}-{timepoint}_camp-pskt'
            fname = f'{stub}-{condition}-fft-ave.fif'
            evoked_spect = mne.read_evokeds(os.path.join(fft_dir, fname))
            assert len(evoked_spect) == 1
            evoked = evoked_spect[0]
            # convert to magnitude/phase
            freqs = evoked.times
            magnitudes = np.abs(evoked.data)
            phases = np.angle(evoked.data)
            # compute the sensor-space SNR
            snr = div_by_adj_bins(magnitudes)
            # prep figure
            these_freqs = (2, 3, 4, 5, 6, 11, 12)
            n_freqs = len(these_freqs)
            n_columns = 5
            xticks = np.arange(freqs[0], freqs[-1], 2)
            fig = plt.figure(figsize=(4 * n_columns, 4 * n_freqs))
            for ix, freq in enumerate(these_freqs):
                # focus on the freq bin we care about
                bin_idx = np.argmin(np.abs(freqs - freq))
                these_phases = phases[..., bin_idx]
                these_magnitudes = magnitudes[..., bin_idx]

                # find the sensor that best captures {freq} Hz activity
                best_sensor = np.argmax(snr[..., bin_idx], axis=0)
                best_sensor_name = evoked.ch_names[bin_idx]

                # spectrum at best sensor
                ax = plt.subplot(n_freqs, n_columns, n_columns * ix + 1)
                ax.plot(freqs, magnitudes[best_sensor].T, linewidth=1)
                ax.set(xticks=xticks, xlabel='freq (Hz)', ylabel='amplitude',
                       title=f'best {freq} Hz amplitude:\n{best_sensor_name}')

                # "SNR" spectrum at best sensor
                ax = plt.subplot(n_freqs, n_columns, n_columns * ix + 2)
                ax.plot(freqs, snr[best_sensor].T, linewidth=1)
                ax.set(xticks=xticks, xlabel='freq (Hz)', ylabel='SNR')
                title = f'best {freq} Hz SNR: {best_sensor_name}'
                if not ix:
                    title = (f'Spectrum amplitude divided by sum of 2 bins on '
                             f'each side\n(channel w/ {title})\n')
                ax.set(title=title)

                # phase vs magnitude (polar)
                ax = plt.subplot(n_freqs, n_columns, n_columns * ix + 3,
                                 polar=True)
                ax.plot(these_phases, these_magnitudes,  '.', alpha=0.5)
                ax.set(ylim=(0, magnitudes.max()))
                # goodness of fit
                estimates = evoked.data[:, bin_idx]
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
                    ax.set(title=('phase vs magnitude with first\n'
                                  'eigenvector and goodness-of-fit'))

                # phase histogram
                ax = plt.subplot(n_freqs, n_columns, n_columns * ix + 4,
                                 polar=True)
                vals, _, _ = ax.hist(these_phases, bins=144)
                ax.set(ylim=(0, vals.max()))
                if not ix:
                    ax.set(title='phase histogram (2.5° bins)\n')

                # weighted phase histogram
                ax = plt.subplot(n_freqs, n_columns, n_columns * ix + 5,
                                 polar=True)
                vals, _, _ = ax.hist(these_phases, bins=144,
                                     weights=these_magnitudes)
                ax.set(ylim=(0, vals.max()))
                if not ix:
                    ax.set(title='weighted phase histogram (2.5° bins)\n')

            fig.suptitle(subj, size=16)
            fig.tight_layout()
            fig.subplots_adjust(top=0.95, left=0.05, hspace=0.5)
            fname = f'{stub}-{condition}-phases.pdf'
            fig.savefig(os.path.join(fig_dir, fname))
            plt.close('all')
