#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot frequency-domain STCs.
"""

import os
from glob import glob
import mne
from aux_functions import load_paths, load_params

mne.cuda.init_cuda()
mne.viz.use_3d_backend('pyvista')

# config paths
data_root, subjects_dir, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'stc', 'morphed-to-fsaverage')
stc_dir = os.path.join(results_dir, 'pskt', 'brain', 'stc')
fig_dir = os.path.join(results_dir, 'pskt', 'brain', 'fig')
for _dir in (stc_dir, fig_dir):
    os.makedirs(_dir, exist_ok=True)

# load params
brain_plot_kwargs, movie_kwargs, subjects = load_params()

# config other
timepoints = ('pre', 'post')

# container for group-level average

# loop over timepoints
for timepoint in timepoints:
    # loop over subjects
    for s in subjects:
        # load this subject's STCs as a generator
        pattern = f'{s}FSAverage-{timepoint}_camp-pskt-[01][0-9]-[lr]h.stc'
        fnames = glob(os.path.join(in_dir, f'{timepoint}_camp', s, pattern))
        stcs = (mne.read_source_estimate(f) for f in fnames)
        # compute average PSD STC
        avg_psd = 0.
        for stc in stcs:
            avg_psd += stc.data
        avg_psd /= len(fnames)
        # use the last STC of the generator to hold the averaged data
        stc.data = avg_psd
        fname = f'{s}-{timepoint}_camp-pskt-avg'
        stc.save(os.path.join(stc_dir, fname))
        # plot it
        brain = stc.plot(subject='fsaverage', **brain_plot_kwargs)
        for freq in (2, 4, 6, 12, 16):
            brain.set_time(freq)
            fname = f'{s}-{timepoint}_camp-pskt-avg-{freq}_Hz.png'
            fpath = os.path.join(fig_dir, fname)
            brain.save_image(fpath)
        del brain
