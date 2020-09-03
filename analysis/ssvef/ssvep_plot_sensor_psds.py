#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Load SSVEP epochs and plot PSDs / scalp topographies for each subject.
"""

import os
import matplotlib.pyplot as plt
import mne
from analysis.aux_functions import load_paths, load_params, load_psd_params

# flags
plot_psd = True
plot_topo = True
cohorts = 'all'

# config paths
data_root, subjects_dir, results_dir = load_paths(cohorts=cohorts)
in_dir = os.path.join(results_dir, 'pskt', 'epochs')
fig_dir = os.path.join(results_dir, 'pskt', 'fig', 'psd')
topo_dir = os.path.join(results_dir, 'pskt', 'fig', 'topo')
for _dir in (fig_dir, topo_dir):
    os.makedirs(_dir, exist_ok=True)

# load params
_, _, subjects = load_params(cohorts=cohorts)
# psd_params = load_psd_params()

# config other
timepoints = ('pre', 'post')
# bandwidth = psd_params['bandwidth']
# subdivide_epochs = psd_params['epoch_dur']
bandwidth = 0.20
subdivide_epochs = 5
subdiv = f'-{subdivide_epochs}_sec' if subdivide_epochs else ''

# loop over subjects
for s in subjects:
    # loop over timepoints
    for timepoint in timepoints:
        stub = f'{s}-{timepoint}_camp-pskt{subdiv}'
        # load epochs (TODO: separately for "ps" and "kt" trials)
        fname = f'{stub}-epo.fif'
        epochs = mne.read_epochs(os.path.join(in_dir, fname), proj=True)
        # plot the sensor-space PSD
        if plot_psd:
            fig = epochs.plot_psd(fmin=0, fmax=20, bandwidth=bandwidth)
            fig.axes[0].set_xticks(range(0, 21, 2))
            fname = f'{stub}-sensor_psd.pdf'
            fig.savefig(os.path.join(fig_dir, fname))
        # plot topomap
        if plot_topo:
            bands = [(freq, f'{freq} Hz') for freq in (2, 4, 6, 12)]
            fig, axs = plt.subplots(2, len(bands), figsize=(10, 4))
            for row, ch_type in zip(axs, ('mag', 'grad')):
                epochs.plot_psd_topomap(bands, bandwidth=bandwidth,
                                        ch_type=ch_type, vlim='joint',
                                        axes=row)
            fig.suptitle(f'Power-normalized field maps ({bandwidth} Hz '
                         'multitaper bandwidth)\ntop: mags; bottom: grads')
            fname = f'{stub}-psd_topomap.pdf'
            fig.savefig(os.path.join(topo_dir, fname))
        plt.close('all')
