#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Extract average time courses in a given label, across experimental conditions
and subjects.
"""

import os
import mne
from mayavi import mlab
from matplotlib.colors import to_rgba
from aux_functions import (load_paths, load_params, get_dataframe_from_label,
                           plot_label, plot_label_and_timeseries)

mlab.options.offscreen = True
mne.cuda.init_cuda()
n_jobs = 10

# load params
brain_plot_kwargs, movie_kwargs, subjects = load_params()

# config paths
data_root, subjects_dir, results_dir = load_paths()
roi_dir = os.path.join('..', 'ROIs')
timeseries_dir = os.path.join(results_dir, 'roi', 'time-series')
img_dir = os.path.join(results_dir, 'roi', 'images')
for dir in (timeseries_dir, img_dir):
    os.makedirs(dir, exist_ok=True)

# load fsaverage source space
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)
fsaverage_src = mne.add_source_space_distances(fsaverage_src, dist_limit=0)

# load the ROI labels
rois = dict()
roi_colors = list('yrgbcm')
for region_number in range(6):
    fpath = os.path.join(roi_dir, f'ventral_ROI_{region_number}-lh.label')
    label = mne.read_label(fpath)
    label.subject = 'fsaverage'
    label.color = to_rgba(roi_colors[region_number])
    rois[region_number] = label


all_conditions = ('words', 'faces', 'cars')
all_timepoints = ('post', 'pre')
methods = ('dSPM', 'sLORETA')
group_lists = (['grandavg'], ['letter', 'language'], ['upper', 'lower'])

for region_number, label in rois.items():
    # prepare to plot
    lineplot_kwargs = dict(hue='condition', hue_order=all_conditions,
                           style='timepoint', style_order=all_timepoints)
    # plot
    for method in methods:
        # get dataframe
        df = get_dataframe_from_label(label, fsaverage_src, methods=[method])
        for groups in group_lists:
            # plot label
            group_str = 'Versus'.join([g.capitalize() for g in groups])
            img_fname = f'{method}-{group_str}-roi-{region_number}.png'
            img_path = os.path.join(img_dir, img_fname)
            plot_label(label, img_path, **brain_plot_kwargs)
            # plot timeseries
            plot_label_and_timeseries(label, img_path, df, method, groups,
                                      timepoints=all_timepoints,
                                      conditions=all_conditions,
                                      all_timepoints=all_timepoints,
                                      all_conditions=all_conditions,
                                      cluster=None,
                                      lineplot_kwargs=lineplot_kwargs)
    # save dataframe
    df.to_csv(os.path.join(timeseries_dir,
                           f'roi-{region_number}-timeseries-long.csv'))
