#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Extract average time courses in a given label, across experimental conditions
and subjects.
"""

import os
import numpy as np
import mne
from matplotlib import rcParams
from matplotlib.colors import to_rgba
from sswef_helpers.aux_functions import (
    load_paths, load_params, load_inverse_params, load_fsaverage_src,
    get_dataframe_from_label, plot_label, plot_label_and_timeseries)

# flags
mne.cuda.init_cuda()
n_jobs = 10

# load params
brain_plot_kwargs, _, subjects, cohort = load_params(experiment='erp')
for kwarg in ('time_viewer', 'show_traces'):
    del brain_plot_kwargs[kwarg]  # not used in Brain.__init__
inverse_params = load_inverse_params()
method = inverse_params['method']
chosen_constraints = ('{orientation_constraint}-{estimate_type}'
                      ).format_map(inverse_params)

# config paths
data_root, subjects_dir, results_dir = load_paths()
roi_dir = os.path.join('..', 'ROIs')
timeseries_dir = os.path.join(results_dir, 'roi', 'time-series')
img_dir = os.path.join(results_dir, 'roi', 'images')
for dir in (timeseries_dir, img_dir):
    os.makedirs(dir, exist_ok=True)

# load fsaverage source space
fsaverage_src = load_fsaverage_src()

rois = dict()
roi_colors = rcParams['axes.prop_cycle'].by_key()['color']
snr_thresholds = np.linspace(1.5, 2.5, 11)

# load anatomical ROI labels
for region_number in range(6):
    fpath = os.path.join(roi_dir, f'ventral_ROI_{region_number}-lh.label')
    label = mne.read_label(fpath)
    label.subject = 'fsaverage'
    label.color = to_rgba(roi_colors[region_number])
    rois[f'ventral_{region_number}'] = label
# load SNR-based labels
for freq in (4,):
    for ix, snr in enumerate(snr_thresholds):
        slug = f'{freq}_Hz-SNR_{snr:.1f}'
        fname = f'{chosen_constraints}-{slug}-lh.label'
        fpath = os.path.join(roi_dir, fname)
        label = mne.read_label(fpath)
        # include only non-empty labels
        if len(label.values):
            label.subject = 'fsaverage'
            label.color = to_rgba(roi_colors[ix % len(roi_colors)])
            rois[slug] = label
# add custom labels
fnames = ('2Hz_LetterKnowledge.lh.label',
          'MPM_IOS_IOG_lh.label',
          'MPM_pOTS_lh.label')
for fname in fnames:
    fpath = os.path.join(roi_dir, fname)
    label = mne.read_label(fpath)
    key = fname.split('.')[0]
    rois[key] = label
# combine the two MPM labels
rois['MPM_IOS_IOG_pOTS_lh'] = (rois['MPM_IOS_IOG_lh'] +
                               rois['MPM_pOTS_lh'])

all_conditions = ('words', 'faces', 'cars')
all_timepoints = ('post', 'pre')
if cohort == 'replication':
    group_lists = (['grandavg'], ['letter'], ['upper', 'lower'])
else:
    group_lists = (['grandavg'], ['letter', 'language'], ['upper', 'lower'])

for region, label in rois.items():
    # prepare to plot
    lineplot_kwargs = dict(hue='condition', hue_order=all_conditions,
                           style='timepoint', style_order=all_timepoints)
    # get dataframe
    df = get_dataframe_from_label(label, fsaverage_src, experiment='erp')
    df['roi'] = region
    # plot
    for groups in group_lists:
        # plot label
        group_str = 'Versus'.join([g.capitalize() for g in groups])
        img_fname = f'{method}-{group_str}-roi-{region}.png'
        img_path = os.path.join(img_dir, img_fname)
        plot_label(label, img_path, **brain_plot_kwargs)
        # plot timeseries
        this_df = df.loc[df['method'] == method]
        plot_label_and_timeseries(label, img_path, this_df, method, groups,
                                  timepoints=all_timepoints,
                                  conditions=all_conditions,
                                  all_timepoints=all_timepoints,
                                  all_conditions=all_conditions,
                                  cluster=None,
                                  lineplot_kwargs=lineplot_kwargs)
    # save dataframe
    df.to_csv(os.path.join(timeseries_dir,
                           f'roi-{region}-timeseries-long.csv'))
