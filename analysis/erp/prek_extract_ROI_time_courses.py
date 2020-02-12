#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot movies with significant cluster regions highlighted.
"""

import os
import numpy as np
import pandas as pd
import mne
from aux_functions import load_paths, load_params, load_cohorts

n_jobs = 10

# load params
brain_plot_kwargs, movie_kwargs, subjects = load_params()

# config paths
data_root, subjects_dir, results_dir = load_paths()
roi_dir = os.path.join('..', 'ROIs')
timeseries_dir = os.path.join(results_dir, 'roi', 'time-series')
os.makedirs(timeseries_dir, exist_ok=True)

# load cohort info (keys Language/LetterIntervention and Lower/UpperKnowledge)
intervention_group, letter_knowledge_group = load_cohorts()

# variables to loop over; subtractions between conditions are (lists of) tuples
methods = ('dSPM', 'sLORETA')
timepoints = ('preCamp', 'postCamp')
conditions = ['words', 'faces', 'cars', 'aliens']

# load fsaverage source space
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)

# load the ROI labels
rois = dict()
roi_colors = list('-rgbcm')  # index 0 won't be used
for region_number in range(1, 6):
    fpath = os.path.join(roi_dir, f'ventral_band_{region_number}-lh.label')
    label = mne.read_label(fpath)
    label.subject = 'fsaverage'
    label = label.restrict(fsaverage_src)
    rois[region_number] = label


time_courses = dict()
# loop over source localization algorithms
for method in methods:
    time_courses[method] = dict()
    # loop over ROIs
    for roi, label in rois.items():
        time_courses[method][roi] = dict()
        # loop over pre/post measurement time
        for timept in timepoints:
            time_courses[method][roi][timept] = dict()
            prepost = timept.rstrip('Camp')
            # loop over conditions
            for cond in conditions:
                time_courses[method][roi][timept][cond] = dict()
                # loop over all subjects
                for s in subjects:
                    # load STC
                    this_subj = os.path.join(data_root, f'{prepost}_camp',
                                             'twa_hp', 'erp', s)
                    fname = f'{s}FSAverage_{prepost}Camp_{method}_{cond}'
                    stc_path = os.path.join(this_subj, 'stc', fname)
                    stc = mne.read_source_estimate(stc_path)
                    # extract label time course
                    timecourse = mne.extract_label_time_course(
                        stc, label, src=fsaverage_src, mode='pca_flip',
                        allow_empty=True)
                    time_courses[method][roi][timept][cond][s] = \
                        np.squeeze(timecourse)
                # convert dict of each subj's time series to DataFrame
                df = pd.DataFrame(time_courses[method][roi][timept][cond],
                                  index=range(len(stc.times)))
                df['time'] = stc.times
                df['condition'] = cond
                time_courses[method][roi][timept][cond] = df
            # combine DataFrames across conditions
            dfs = (time_courses[method][roi][timept][c] for c in conditions)
            time_courses[method][roi][timept] = pd.concat(dfs)
            time_courses[method][roi][timept]['timepoint'] = timept
        # combine DataFrames across timepoints
        dfs = (time_courses[method][roi][t] for t in timepoints)
        time_courses[method][roi] = pd.concat(dfs)
        time_courses[method][roi]['roi'] = roi
    # combine DataFrames across ROIs
    dfs = (time_courses[method][r] for r in rois)
    time_courses[method] = pd.concat(dfs)
    time_courses[method]['method'] = method
# combine DataFrames across methods
dfs = (time_courses[m] for m in methods)
time_courses = pd.concat(dfs)

# reshape DataFrame
all_cols = time_courses.columns.values
subj_cols = time_courses.columns.str.startswith('prek')
id_vars = all_cols[np.logical_not(subj_cols)]
df = pd.melt(time_courses, id_vars=id_vars, var_name='subj')
# add columns for cohort
intervention_map = {subj: group.lower().rstrip('intervention')
                    for group, members in intervention_group.items()
                    for subj in members}
knowledge_map = {subj: group.lower().rstrip('knowledge')
                 for group, members in letter_knowledge_group.items()
                 for subj in members}
df['intervention'] = df['subj'].map(intervention_map)
df['pretest'] = df['subj'].map(knowledge_map)
# save
time_courses.to_csv(os.path.join(timeseries_dir, 'roi-timeseries-wide.csv'))
df.to_csv(os.path.join(timeseries_dir, 'roi-timeseries-long.csv'))
