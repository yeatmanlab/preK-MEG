#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Extract SSVEP epochs, downsample, and save to disk. Optionally generate
spectrum estimates and plot PSDs / scalp topographies for each subject.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from aux_functions import load_paths, load_params

mne.cuda.init_cuda()

# flags
compute_psds = True
plot_psds = True
plot_topomaps = True
n_jobs = 10

# config paths
data_root, subjects_dir, results_dir = load_paths()
epo_dir = os.path.join(results_dir, 'pskt', 'epochs')
fig_dir = os.path.join(results_dir, 'pskt', 'spectra', 'figs')
psd_dir = os.path.join(results_dir, 'pskt', 'spectra', 'psds')
for _dir in (epo_dir, fig_dir, psd_dir):
    os.makedirs(_dir, exist_ok=True)

# load params
_, _, subjects = load_params()

# config other
timepoints = ('pre', 'post')
runs = (1, 2)
trial_dur = 20  # seconds
# reject_dict = dict(mag=8000e-15,   # 8000 fT
#                    grad=8000e-13)  # 8000 fT/cm

# loop over subjects
for s in subjects:
    for timepoint in timepoints:
        this_subj = os.path.join(data_root, f'{timepoint}_camp', 'twa_hp',
                                 'pskt', s)
        raws = list()
        events_list = list()
        first_samps = list()
        last_samps = list()
        # extract the events from the original raw (STIM channels are dropped
        # from the preprocessed raw) TODO: this will prob. change at some point
        for run in runs:
            this_fname = f'{s}_pskt_{run:02}_{timepoint}_raw.fif'
            raw_path = os.path.join(this_subj, 'raw_fif', this_fname)
            raw = mne.io.read_raw_fif(raw_path, allow_maxshield=True)
            eve = mne.find_events(raw, stim_channel='STI001')
            eve[3:, 2] = 7  # fix events to code for PS vs KT differently
            first_samps.append(raw.first_samp)
            last_samps.append(raw.last_samp)
            events_list.append(eve)
        events = mne.concatenate_events(events_list, first_samps, last_samps)
        # now process the clean files
        for run in runs:
            this_fname = f'{s}_pskt_{run:02}_{timepoint}_allclean_fil80_raw_sss.fif'  # noqa E501
            raw_path = os.path.join(this_subj, 'sss_pca_fif', this_fname)
            raw = mne.io.read_raw_fif(raw_path)
            raws.append(raw)
        # combine runs
        raw, events = mne.concatenate_raws(raws, events_list=events_list)
        # downsample
        raw, events = raw.resample(sfreq=50, events=events, n_jobs='cuda')
        # clean up
        del raws, first_samps, last_samps, events_list
        # epoch
        event_dict = dict(ps=5, kt=7)
        epochs = mne.Epochs(raw, events, event_dict, tmin=0, tmax=trial_dur,
                            baseline=None, proj=True,  # reject=reject_dict,
                            reject_by_annotation=False, preload=True)
        # save epochs
        fname = f'{s}-{timepoint}_camp-pskt-epo.fif'
        epochs.save(os.path.join(epo_dir, fname), fmt='double')
        # PSD settings
        psd_kwargs = dict(fmin=0, fmax=20, bandwidth=0.1, adaptive=False,
                          n_jobs=n_jobs)
        # compute PSDs
        if compute_psds:
            psds, freqs = mne.time_frequency.psd_multitaper(epochs,
                                                            **psd_kwargs)
            fname = f"{s}-{timepoint}_camp-pskt-{psd_kwargs['bandwidth']}Hz.npz"  # noqa E501
            np.savez(os.path.join(psd_dir, fname), psds=psds, freqs=freqs)
        # plot
        average = False
        ave = 'average-' if average else ''
        if plot_psds:
            fig = epochs.plot_psd(average=average, **psd_kwargs)
            fig.axes[0].set_xticks(range(0, 21, 2))
            fig_name = f"{s}-{timepoint}-bw{psd_kwargs['bandwidth']}-{ave}PSKT.png"  # noqa E501
            fig.savefig(os.path.join(fig_dir, fig_name))
            plt.close(fig)
        if plot_topomaps:
            # make bands just big enough to get 1 freq bin
            bin_spacing = np.diff(freqs)[0]
            bin_width = np.array([0, bin_spacing]) - bin_spacing / 2
            bands = list()
            for freq in (2, 4, 6, 12, 16):
                bin_freq = freqs[np.argmin(np.abs(freqs - freq))]
                band = tuple(bin_width + bin_freq)
                bands.append(band + (f'{freq} Hz',))
            bands.append((7, 9, 'mystery bump\n(7-9 Hz)'))
            # plot
            fig, axs = plt.subplots(2, 6, figsize=(15, 5))
            for row, ch_type in zip(axs, ('mag', 'grad')):
                epochs.plot_psd_topomap(bands, bandwidth=0.1, ch_type=ch_type,
                                        normalize=True, axes=row,
                                        cmap='inferno')
            fig.suptitle('Power-normalized field maps (0.1 Hz multitaper '
                         'bandwidth)\ntop: mags; bottom: grads')
            fig_name = f'{s}-{timepoint}-topomaps.png'
            fig.savefig(os.path.join(fig_dir, fig_name))
            plt.close(fig)
