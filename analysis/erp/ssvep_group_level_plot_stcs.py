#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot frequency-domain STCs.
"""

import os
import numpy as np
from scipy.ndimage import convolve1d
from mayavi import mlab
import mne
from aux_functions import load_paths, load_params, load_cohorts

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
kinds = ('baseline', 'phase_cancelled')
trial_dur = 20
subdivide_epochs = 5
subdiv = f'-{subdivide_epochs}_sec' if subdivide_epochs else ''

# load in all data first, to get colormap limits. Not memory efficient but
# shouldn't be too bad
all_data = list()
all_stcs = dict()
for s in groups['GrandAvg']:
    for timepoint in timepoints:
        fname = f'{s}FSAverage-{timepoint}_camp-pskt{subdiv}-fft-stc.h5'
        stc = mne.read_source_estimate(os.path.join(in_dir, fname),
                                       subject='fsaverage')
        all_stcs[f'{s}-{timepoint}'] = stc
        all_data.append(stc.data)
all_data = np.array(all_data)
lims = tuple(np.percentile(np.abs(all_data), (90, 99, 99.9)))
clim = dict(kind='value', lims=lims)

# loop over timepoints
for timepoint in timepoints:
    # loop over cohort groups
    for group, members in groups.items():
        # only do pretest knowledge comparison for pre-camp timepoint
        if group.endswith('Knowledge') and timepoint == 'post':
            continue

        # aggregate over group members
        data = 0.
        for s in members:
            data += np.abs(all_stcs[f'{s}-{timepoint}'].data)
        # use a copy of the last STC as container
        avg_stc = all_stcs[f'{s}-{timepoint}'].copy()
        avg_stc.data = data

        # divide each bin by its two neighbors on each side to get "SNR"
        snr_stc = avg_stc.copy()
        weights = [0.25, 0.25, 0, 0.25, 0.25]
        snr_stc.data = data / convolve1d(data, mode='constant',
                                         weights=weights)

        for kind, _stc in zip(['avg', 'snr'], [avg_stc, snr_stc]):
            # save stc
            fname = f'{group}-{timepoint}_camp-pskt{subdiv}-fft-{kind}'
            stc.save(os.path.join(stc_dir, fname), ftype='h5')
            # plot stc
            brain = stc.plot(subject='fsaverage', clim=clim,
                             **brain_plot_kwargs)
            for freq in (2, 4, 6, 12):
                brain.set_time(freq)
                fpath = os.path.join(fig_dir, f'{fname}-{freq:02}_Hz.png')
                brain.save_image(fpath)
            del brain
