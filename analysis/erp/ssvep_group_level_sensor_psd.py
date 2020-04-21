#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Compute and plot group-level PSDs and topomaps.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from aux_functions import load_paths, load_params, load_cohorts

mne.cuda.init_cuda()

# flags
compute_psds = True
plot_psds = True
plot_topomaps = True
n_jobs = 10

# config paths
data_root, subjects_dir, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'epochs')
epo_dir = os.path.join(results_dir, 'pskt', 'group-level', 'epochs')
fig_dir = os.path.join(results_dir, 'pskt', 'group-level', 'spectra', 'figs')
psd_dir = os.path.join(results_dir, 'pskt', 'group-level', 'spectra', 'psds')
for _dir in (epo_dir, fig_dir, psd_dir):
    os.makedirs(_dir, exist_ok=True)

# load params
_, _, subjects = load_params()
intervention_group, letter_knowledge_group = load_cohorts()
groups = dict(GrandAvg=subjects)
groups.update(intervention_group)
groups.update(letter_knowledge_group)

# config other
timepoints = ('pre', 'post')

# PSD settings
psd_kwargs = dict(fmin=0, fmax=20, bandwidth=0.1, adaptive=False,
                  n_jobs=n_jobs)

dev_head_t = None

# loop over subjects
for timepoint in timepoints:
    for group, members in groups.items():
        # only do pretest knowledge comparison for pre-camp timepoint
        if group.endswith('Knowledge') and timepoint == 'post':
            continue
        # only do intervention cohort comparison for post-camp timepoint
        if group.endswith('Intervention') and timepoint == 'pre':
            continue

        # filename stub
        stub = f"{group}-{timepoint}_camp-pskt"

        # load and combine epochs
        epochs_list = list()
        for subj in members:
            fname = f'{subj}-{timepoint}_camp-pskt-epo.fif'
            fpath = os.path.join(in_dir, fname)
            epochs = mne.read_epochs(fpath, proj=True)
            epochs.info['projs'] = list()    # hack
            epochs_list.append(epochs)
            # hack to allow combining epochs from different head pos:
            if dev_head_t is None:
                dev_head_t = epochs.info['dev_head_t']
            else:
                epochs.info['dev_head_t'] = dev_head_t
        # concatenate and save epochs
        epochs = mne.concatenate_epochs(epochs_list)
        epochs.save(os.path.join(epo_dir, f'{stub}_epo.fif'), fmt='double')

        # compute PSDs
        psds, freqs = mne.time_frequency.psd_multitaper(epochs, **psd_kwargs)
        psd_fname = f"{stub}-psd_{psd_kwargs['bandwidth']}_Hz.npz"
        np.savez(os.path.join(psd_dir, psd_fname), psds=psds, freqs=freqs)

        # plot PSD
        average = False
        ave = 'average-' if average else ''
        if plot_psds:
            fig = epochs.plot_psd(average=average, **psd_kwargs)
            fig.axes[0].set_xticks(range(0, 21, 2))
            fig.savefig(os.path.join(fig_dir, f'{stub}_psd.png'))
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

            fig.savefig(os.path.join(fig_dir, f'{stub}_topomaps.png'))
            plt.close(fig)
