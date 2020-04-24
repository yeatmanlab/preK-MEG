#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Load SSVEP epochs and plot PSDs, phases, etc.

data at https://dan.mccloy.info/data/prek_1964-pre_camp-pskt-epo.fif
"""

import os
import numpy as np
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
import mne
from mne.time_frequency.multitaper import (_compute_mt_params, _mt_spectra,
                                           _psd_from_mt)
from aux_functions import load_paths, load_params

# config paths
data_root, subjects_dir, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'epochs')
fig_dir = os.path.join(results_dir, 'pskt', 'fig', 'phase')
os.makedirs(fig_dir, exist_ok=True)

# load params
_, _, subjects = load_params()

# config other
timepoints = ('pre', 'post')


# loop over subjects
for subj in subjects:
    # loop over timepoints
    for timepoint in timepoints:
        # load epochs
        stub = f'{subj}-{timepoint}_camp-pskt'
        epochs = mne.read_epochs(os.path.join(in_dir, f'{stub}-epo.fif'),
                                 proj=True)
        # cut off last sample
        epochs.crop(None, epochs.times[-2])
        assert epochs.times.size == 1000

        for div, bandwidth in zip((1, 2, 4, 5, 10), (0.1, 0.1, 0.2, 0.5, 1.)):
            # reshape epochs data to get different numbers of epochs
            data = epochs.get_data()
            n_epochs, n_channels, n_times = data.shape
            assert n_times % div == 0
            new_n_times = n_times // div
            new_shape = (n_epochs, n_channels, div, new_n_times)
            data = np.reshape(data, new_shape)
            data = data.transpose(0, 2, 1, 3)
            data = np.reshape(data, (div * n_epochs, n_channels, new_n_times))
            recut_epochs = mne.EpochsArray(data, epochs.info)
            evoked = recut_epochs.average()
            assert len(evoked.times) == new_n_times

            # do multitaper estimation
            sfreq = evoked.info['sfreq']
            mt_kwargs = dict(n_times=new_n_times, sfreq=sfreq,
                             bandwidth=bandwidth, low_bias=True,
                             adaptive=False)
            dpss, eigvals, adaptive = _compute_mt_params(**mt_kwargs)
            assert dpss.shape[0] == 1  # otherwise plotting will fail
            n_fft = new_n_times  # _mt_spectra defaults to wrong axis
            mt_spectra, freqs = _mt_spectra(evoked.data, dpss, sfreq,
                                            n_fft=n_fft)
            magnitudes = np.abs(mt_spectra)
            phases = np.angle(mt_spectra)
            # compute the sensor-space PSD
            sensor_weights = np.sqrt(eigvals)[np.newaxis, :, np.newaxis]
            sensor_psd = _psd_from_mt(mt_spectra, sensor_weights)
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
                these_phases = np.squeeze(phases[..., bin_idx])
                these_magnitudes = np.squeeze(magnitudes[..., bin_idx])

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
                spectrum = mt_spectra[:, 0, bin_idx]
                spectrum_ri = np.array((spectrum.real, spectrum.imag))
                u, s, v = np.linalg.svd(spectrum_ri, full_matrices=False)
                gof = 100 * s[0] / s.sum()
                # plot angle of first eigen & GOF
                u = u[:, 0] * s[0]
                u = u[0] + 1j * u[1]
                ax.plot([0, np.angle(u)], [0, np.abs(u)], color='k', zorder=2)
                ax.text(0, 0, f'{gof:0.1}%', style='plain')
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
            fname = f'{stub}-{20 // div}_sec-phases.pdf'
            fig.savefig(os.path.join(fig_dir, fname))
            plt.close('all')
