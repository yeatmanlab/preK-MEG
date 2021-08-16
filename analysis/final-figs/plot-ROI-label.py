#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot a cortical label for a figure.
"""

import os
import mne
from analysis.aux_functions import load_paths, load_params

# config paths
data_root, subjects_dir, results_dir = load_paths()
roi_dir = os.path.join('..', 'ROIs')

# config other
brain_plot_kwargs, *_ = load_params()
for kwarg in ('time_viewer', 'show_traces'):
    del brain_plot_kwargs[kwarg]  # not used in Brain.__init__
brain_plot_kwargs.update(views='ven', hemi='lh', surf='inflated',
                         background='white', size=(2400, 4800))


# load fsaverage source space
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)
fsaverage_src = mne.add_source_space_distances(fsaverage_src, dist_limit=0)

# load the labels
fnames = ('MPM_IOS_IOG_lh.label', 'MPM_pOTS_lh.label')
labels = [mne.read_label(os.path.join(roi_dir, fname)) for fname in fnames]
for lab in labels:
    lab.values.fill(1.)
    lab.smooth(subject='fsaverage', subjects_dir=subjects_dir, smooth=1)
# combine the two MPM labels
label = labels[0] + labels[1]

# plot the label
img_fname = 'MPM_IOS_IOG_pOTS_lh.png'
img_path = os.path.join('.', img_fname)

brain = mne.viz.Brain('fsaverage', **brain_plot_kwargs)
brain.add_label(label, alpha=0.75, color='#44BB99')  # EE7733 → orange
brain.add_label(label, alpha=1, color='#44BB99', borders=True)
brain.show_view(dict(azimuth=240, elevation=150), roll=0)
brain.plotter.disable_3_lights()
brain.save_image(img_path)
