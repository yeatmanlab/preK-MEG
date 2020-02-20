#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Extract average time courses in a given label, across experimental conditions
and subjects.
"""

import os
import mne
from aux_functions import load_paths, load_params, get_dataframe_from_label

n_jobs = 10
use_ventral_band_rois = False

# load params
brain_plot_kwargs, movie_kwargs, subjects = load_params()

# config paths
data_root, subjects_dir, results_dir = load_paths()
roi_dir = os.path.join('..', 'ROIs')
timeseries_dir = os.path.join(results_dir, 'roi', 'time-series')
os.makedirs(timeseries_dir, exist_ok=True)

# load fsaverage source space
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)

# load the ROI labels
rois = dict()
roi_colors = list('yrgbcm')
if use_ventral_band_rois:
    for region_number in range(1, 6):
        fpath = os.path.join(roi_dir, f'ventral_band_{region_number}-lh.label')
        label = mne.read_label(fpath)
        label.subject = 'fsaverage'
        label = label.restrict(fsaverage_src)
        rois[region_number] = label
else:
    fpath = os.path.join(roi_dir, 'ventral_ROI-lh.label')
    label = mne.read_label(fpath)
    label.subject = 'fsaverage'
    label = label.restrict(fsaverage_src)
    rois[0] = label

df = get_dataframe_from_label(label, fsaverage_src)

# save
bands = '-bands' if use_ventral_band_rois else ''
df.to_csv(os.path.join(timeseries_dir, f'roi{bands}-timeseries-long.csv'))
