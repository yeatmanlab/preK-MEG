#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot uncorrected t-value maps (of 2 Hz pre_camp) at various threshold levels.
"""

import os
import numpy as np
from mayavi import mlab
import mne
from aux_functions import load_paths, load_params

mlab.options.offscreen = True
mne.cuda.init_cuda()

# flags
save_movie = True

# config paths
data_root, subjects_dir, results_dir = load_paths()
stc_dir = os.path.join(results_dir, 'pskt', 'group-level', 'stc')
tval_dir = os.path.join(results_dir, 'pskt', 'group-level', 'tvals')
fig_dir = os.path.join(results_dir, 'pskt', 'group-level', 'fig', 'thresholds')
os.makedirs(fig_dir, exist_ok=True)

# load params
brain_plot_kwargs, movie_kwargs, subjects = load_params()

# load an STC as a template
fname = 'GrandAvg-pre_camp-pskt-5_sec-fft-avg'
stc = mne.read_source_estimate(os.path.join(stc_dir, fname))
all_freqs = stc.times
vertices = stc.vertices
hemi_n_verts = vertices[0].size

# load fsaverage source space
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)

# config other
timepoints = ('pre',)
precamp_fname = 'GrandAvg-pre_camp'
freq = 2
bin_idx = np.argmin(np.abs(all_freqs - freq))

for prefix in (precamp_fname,):
    fname = f'{prefix}-tvals.npy'
    tvals = np.load(os.path.join(tval_dir, fname))
    stc.data = tvals
    # set the colormap lims
    lims = tuple(np.percentile(tvals, (95, 99, 99.9)))
    clim = dict(kind='value', lims=lims)
    # plot the brain
    brain = stc.plot(smoothing_steps='nearest', clim=clim, time_unit='s',
                     time_label='t-value (%0.2f Hz)', initial_time=freq,
                     **brain_plot_kwargs)
    for threshold in np.arange(4, 6.6, 0.5):
        for hemi in ('lh', 'rh'):
            tv = tvals[:hemi_n_verts] if hemi == 'lh' else tvals[hemi_n_verts:]
            verts = np.where(tv[:, bin_idx] >= threshold)[0]
            label = mne.Label(verts, hemi=hemi, subject=stc.subject)
            # fill in verts that are surrounded by cluster verts
            label = label.fill(fsaverage_src)
            brain.add_label(label, borders=False, color='c')
        img_fname = f'{prefix}-{freq:02}_Hz-threshold_{threshold:3.1f}.png'
        img_path = os.path.join(fig_dir, img_fname)
        brain.save_image(img_path)
        brain.remove_labels()
    del brain
