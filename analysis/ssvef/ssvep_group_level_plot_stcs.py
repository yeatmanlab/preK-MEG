#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot frequency-domain STCs.
"""

import os
import numpy as np
from mayavi import mlab
import mne
from analysis.aux_functions import (load_paths, load_params, load_cohorts,
                           div_by_adj_bins)


# flags
mlab.options.offscreen = True
mne.cuda.init_cuda()
mne.viz.set_3d_backend('mayavi')

# config paths
_, _, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'stc', 'morphed-to-fsaverage')
stc_dir = os.path.join(results_dir, 'pskt', 'group-level', 'stc')
fig_dir = os.path.join(results_dir, 'pskt', 'group-level', 'fig', 'brain')
for _dir in (stc_dir, fig_dir):
    os.makedirs(_dir, exist_ok=True)

# load params
brain_plot_kwargs, movie_kwargs, subjects = load_params()
intervention_group, letter_knowledge_group = load_cohorts()
groups = dict(GrandAvg=subjects)
groups.update(intervention_group)
groups.update(letter_knowledge_group)

# config other
timepoints = ('pre', 'post')
trial_dur = 20
subdivide_epochs = 5
subdiv = f'-{subdivide_epochs}_sec' if subdivide_epochs else ''

# load in all individual subject data first, to get colormap limits. Not very
# memory efficient but shouldn't be too bad
all_data = list()
all_stcs = dict()
for s in groups['GrandAvg']:
    for timepoint in timepoints:
        fname = f'{s}FSAverage-{timepoint}_camp-pskt{subdiv}-fft-stc.h5'
        stc = mne.read_source_estimate(os.path.join(in_dir, fname),
                                       subject='fsaverage')
        all_stcs[f'{s}-{timepoint}'] = stc
        all_data.append(stc.data)
abs_data = np.abs(all_data)
snr_data = div_by_adj_bins(abs_data)
# separate lims for untransformed data, SNR, and log(SNR)
cmap_percentiles = (91, 95, 99)
lims = tuple(np.percentile(abs_data, cmap_percentiles))
snr_lims = tuple(np.percentile(snr_data, cmap_percentiles))
log_lims = tuple(np.percentile(np.log(snr_data), cmap_percentiles))
# clean up
del all_data, abs_data, snr_data

# loop over timepoints
for timepoint in timepoints:
    # loop over cohort groups
    for group, members in groups.items():
        # only do pretest knowledge comparison for pre-camp timepoint
        if group.endswith('Knowledge') and timepoint == 'post':
            continue
        # load and plot untransformed data & SNR data
        for kind, _lims in zip(['avg', 'snr', 'log'],
                               [lims, snr_lims, log_lims]):
            # load stc
            fname = f'{group}-{timepoint}_camp-pskt{subdiv}-fft-{kind}'
            fpath = os.path.join(stc_dir, fname)
            stc = mne.read_source_estimate(fpath, subject='fsaverage')
            # plot stc
            clim = dict(kind='value')
            pos = dict(pos_lims=_lims) if kind == 'log' else dict(lims=_lims)
            clim.update(pos)
            brain = stc.plot(subject='fsaverage', clim=clim,
                             **brain_plot_kwargs)
            for freq in (2, 4, 6, 12):
                brain.set_time(freq)
                fpath = os.path.join(fig_dir, f'{fname}-{freq:02}_Hz.png')
                brain.save_image(fpath)
            del brain
