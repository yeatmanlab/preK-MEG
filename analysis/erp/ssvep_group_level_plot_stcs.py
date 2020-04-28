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
from aux_functions import (load_paths, load_params, load_psd_params,
                           load_cohorts)

# flags
mlab.options.offscreen = True
mne.cuda.init_cuda()
mne.viz.set_3d_backend('mayavi')

# config paths
_, _, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'group-level', 'stc')
fig_dir = os.path.join(results_dir, 'pskt', 'group-level', 'fig', 'brain')
os.makedirs(fig_dir, exist_ok=True)

# load params
brain_plot_kwargs, movie_kwargs, subjects = load_params()
psd_params = load_psd_params()
intervention_group, letter_knowledge_group = load_cohorts()
groups = dict(GrandAvg=subjects)
groups.update(intervention_group)
groups.update(letter_knowledge_group)

# config other
timepoints = ('pre', 'post')
kinds = ('baseline', 'phase_cancelled')
trial_dur = 20
divisions = trial_dur // psd_params['epoch_dur']
subdiv = f"-{psd_params['epoch_dur']}_sec" if divisions > 1 else ''

# loop over timepoints
for timepoint in timepoints:
    # loop over cohort groups
    for group, members in groups.items():
        # only do pretest knowledge comparison for pre-camp timepoint
        if group.endswith('Knowledge') and timepoint == 'post':
            continue
        # only do intervention cohort comparison for post-camp timepoint
        if group.endswith('Intervention') and timepoint == 'pre':
            continue

        # preload both STCs so we are assured of the same colormap
        stcs = dict()
        for kind in kinds:
            fname = f'{group}-{timepoint}_camp-pskt{subdiv}-multitaper-{kind}'
            stcs[kind] = mne.read_source_estimate(os.path.join(in_dir, fname))

        data = np.array([s.data for s in stcs])
        lims = tuple(np.percentile(data, 0.95, 0.99, 0.999))
        clim = dict(kind='value', lims=lims)
        # plot it
        for kind, stc in stcs.items():
            brain = stc.plot(subject='fsaverage', clim=clim,
                             **brain_plot_kwargs)
            for freq in (2, 4, 6, 12):
                brain.set_time(freq)
                fpath = os.path.join(fig_dir, f'{fname}-{freq:02}_Hz.png')
                brain.save_image(fpath)
            del brain
