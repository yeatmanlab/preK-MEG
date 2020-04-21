#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot group-average frequency-domain STCs.
"""

import os
from glob import glob
from mayavi import mlab
import mne
from aux_functions import load_paths, load_params, load_cohorts

# flags
mlab.options.offscreen = True
mne.cuda.init_cuda()
mne.viz.set_3d_backend('mayavi')

# config paths
data_root, subjects_dir, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'stc', 'morphed-to-fsaverage')
stc_dir = os.path.join(results_dir, 'pskt', 'group-level', 'brain', 'stc')
fig_dir = os.path.join(results_dir, 'pskt', 'group-level', 'brain', 'fig')
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

# loop over timepoints
for timepoint in timepoints:
    for group, members in groups.items():
        # only do pretest knowledge comparison for pre-camp timepoint
        if group.endswith('Knowledge') and timepoint == 'post':
            continue
        # only do intervention cohort comparison for post-camp timepoint
        if group.endswith('Intervention') and timepoint == 'pre':
            continue

        # filename stub
        stub = f'{group}-{timepoint}_camp-pskt'
        # loop over subjects
        patterns = [os.path.join(in_dir, f'{timepoint}_camp', s,
                                 f'{s}FSAverage-{timepoint}_camp-pskt-[01][0-9]-lh.stc')  # noqa E501
                    for s in members]
        fnames = [g for p in patterns for g in glob(p)]
        stcs = (mne.read_source_estimate(f) for f in fnames)
        # compute average PSD STC
        avg_psd = 0.
        for stc in stcs:
            avg_psd += stc.data
        avg_psd /= len(fnames)
        # use the last STC of the generator to hold the averaged data
        stc.data = avg_psd
        fname = f'{group}-{timepoint}_camp-pskt'
        stc.save(os.path.join(stc_dir, fname))
        # plot it
        brain = stc.plot(subject='fsaverage', **brain_plot_kwargs)
        for freq in (2, 4, 6, 12, 16):
            brain.set_time(freq)
            fname = f'{group}-{timepoint}_camp-pskt-{freq}_Hz.png'
            fpath = os.path.join(fig_dir, fname)
            brain.save_image(fpath)
        del brain
