#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot SNR spectra for a given label
"""

import os
import mne
from analysis.aux_functions import (load_paths, load_params,
                                    load_inverse_params,
                                    get_dataframe_from_label, plot_label,
                                    plot_label_and_timeseries)

# load params
brain_plot_kwargs, _, subjects, cohort = load_params(experiment='pskt')
for kwarg in ('time_viewer', 'show_traces'):
    del brain_plot_kwargs[kwarg]  # not used in Brain.__init__
inverse_params = load_inverse_params()
chosen_constraints = ('{orientation_constraint}-{estimate_type}'
                      ).format_map(inverse_params)

# config paths
data_root, subjects_dir, results_dir = load_paths()
roi_dir = os.path.join('..', 'ROIs')
in_dir = os.path.join(results_dir, 'pskt', 'stc', 'morphed-to-fsaverage')
spectrum_dir = os.path.join(results_dir, 'pskt', 'roi')
img_dir = os.path.join(results_dir, 'pskt', 'fig', 'roi')
for dir in (spectrum_dir, img_dir):
    os.makedirs(dir, exist_ok=True)

# load label
fname = '2Hz_LetterKnowledge.lh.label'
fpath = os.path.join(roi_dir, fname)
region = '2_Hz-LetterKnowledge'
label = mne.read_label(fpath)

# load fsaverage source space
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)
fsaverage_src = mne.add_source_space_distances(fsaverage_src, dist_limit=0)

# config other
methods = ('snr', 'fft')
conditions = ('all', 'ps', 'kt')
timepoints = ('post', 'pre')
if cohort == 'replication':
    group_lists = (['grandavg'], ['letter'], ['upper', 'lower'])
else:
    group_lists = (['grandavg'], ['letter', 'language'], ['upper', 'lower'])

precamp_fname = 'DataMinusNoise1samp-pre_camp'
postcamp_fname = 'DataMinusNoise1samp-post_camp'
median_split_fname = 'UpperVsLowerKnowledge-pre_camp'
intervention_fname = 'LetterVsLanguageIntervention-PostMinusPre_camp'

# prepare to plot
lineplot_kwargs = dict(hue='condition', hue_order=conditions,
                       style='timepoint', style_order=timepoints)

# load brain data & restrict to label
df = get_dataframe_from_label(label, fsaverage_src, methods=methods,
                              conditions=conditions, unit='freq',
                              experiment='pskt')
df['roi'] = region
df.to_csv(os.path.join(spectrum_dir, f'roi-{region}-frequencies-long.csv'))

# plot
for groups in group_lists:
    for method in methods:
        # plot label
        group_str = 'Versus'.join([g.capitalize() for g in groups])
        img_fname = f'{method}-{group_str}-roi-{region}.png'
        img_path = os.path.join(img_dir, img_fname)
        plot_label(label, img_path, **brain_plot_kwargs)
        # plot spectrum
        this_df = df.loc[df['method'] == method]
        plot_label_and_timeseries(label, img_path, this_df, method, groups,
                                  timepoints=timepoints,
                                  conditions=conditions,
                                  all_timepoints=timepoints,
                                  all_conditions=conditions,
                                  cluster=None,
                                  lineplot_kwargs=lineplot_kwargs,
                                  unit='freq')
