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

# loop over timepoints
for timepoint in timepoints:
    # loop over cohort groups
    for group, members in groups.items():
        # only do pretest knowledge comparison for pre-camp timepoint
        if group.endswith('Knowledge') and timepoint == 'post':
            continue

        # aggregation variables
        baseline_data = 0.
        phase_cancelled_data = 0.
        for s in members:
            fname = f'{s}FSAverage-{timepoint}_camp-pskt{subdiv}-fft'
            stc = mne.read_source_estimate(os.path.join(in_dir, fname),
                                           subject='fsaverage')
            phase_cancelled_data += stc.data
            baseline_data += np.abs(stc.data)
        # use the last STC as container
        baseline_stc = stc.copy()
        phase_cancelled_stc = stc.copy()
        del stc
        baseline_stc.data = baseline_data
        phase_cancelled_stc.data = np.abs(phase_cancelled_data)
        # use baseline data to set color scale
        lims = tuple(np.percentile(baseline_data, (90, 95, 99.5)))
        clim = dict(kind='value', lims=lims)
        for kind, stc in zip(['baseline', 'phase_cancelled'],
                             [baseline_stc, phase_cancelled_stc]):
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
