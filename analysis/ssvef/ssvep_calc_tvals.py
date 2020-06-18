#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Compute uncorrected t-value maps.
"""

import os
import numpy as np
import mne
from mne.stats import ttest_ind_no_p, ttest_1samp_no_p
from analysis.aux_functions import (load_paths, load_params, load_cohorts,
                           div_by_adj_bins)

# config paths
data_root, subjects_dir, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'stc', 'morphed-to-fsaverage')
stc_dir = os.path.join(results_dir, 'pskt', 'group-level', 'stc')
fig_dir = os.path.join(results_dir, 'pskt', 'fig', 'tvals')
npz_dir = os.path.join(results_dir, 'pskt', 'group-level', 'npz')
tval_dir = os.path.join(results_dir, 'pskt', 'group-level', 'tvals')
for _dir in (tval_dir, fig_dir, npz_dir):
    os.makedirs(_dir, exist_ok=True)

# load params
brain_plot_kwargs, _, subjects = load_params()
intervention_group, letter_knowledge_group = load_cohorts()
groups = dict(GrandAvg=subjects)
groups.update(intervention_group)
groups.update(letter_knowledge_group)

# config other
timepoints = ('pre', 'post')
subdivide_epochs = 5
subdiv = f'-{subdivide_epochs}_sec' if subdivide_epochs else ''

# load in all the data
data_dict = dict()
noise_dict = dict()
for s in subjects:
    for timepoint in timepoints:
        stub = f'{s}FSAverage-{timepoint}_camp-pskt{subdiv}-fft'
        stc = mne.read_source_estimate(os.path.join(in_dir, f'{stub}-stc.h5'),
                                       subject='fsaverage')
        # compute magnitude (signal) & avg of adjacent bins on either side
        # (noise), & save for later group comparisons
        data_dict[f'{s}-{timepoint}'] = np.abs(stc.data)
        noise_dict[f'{s}-{timepoint}'] = div_by_adj_bins(np.abs(stc.data),
                                                         return_noise=True)
np.savez(os.path.join(npz_dir, 'data.npz'), **data_dict)
np.savez(os.path.join(npz_dir, 'noise.npz'), **noise_dict)

# across-subj 1-sample t-values (freq bin versus mean of 4 surrounding bins)
for tpt in timepoints:
    data = np.array([data_dict[f'{s}-{tpt}'] for s in groups['GrandAvg']])
    noise = np.array([noise_dict[f'{s}-{tpt}'] for s in groups['GrandAvg']])
    tvals = ttest_1samp_no_p(data - noise)
    np.save(os.path.join(tval_dir, f'GrandAvg-{tpt}_camp-tvals.npy'), tvals)

# planned comparison: pre-intervention median split on letter awareness test
median_split = list()
for group in ('UpperKnowledge', 'LowerKnowledge'):
    data = np.array([data_dict[f'{s}-pre'] for s in groups[group]])
    median_split.append(data)
median_split_tvals = ttest_ind_no_p(*median_split)

# planned comparison: post-minus-pre-intervention, language-vs-letter cohort
intervention = list()
for group in ('LetterIntervention', 'LanguageIntervention'):
    data = np.array([data_dict[f'{s}-post'] - data_dict[f'{s}-pre']
                    for s in groups[group]])
    intervention.append(data)
intervention_tvals = ttest_ind_no_p(*intervention)

# save the data
tval_dict = {'UpperVsLowerKnowledge-pre_camp': median_split_tvals,
             'LetterVsLanguageIntervention-PostMinusPre_camp': intervention_tvals}  # noqa E501
for fname, tvals in tval_dict.items():
    np.save(os.path.join(tval_dir, f'{fname}-tvals.npy'), tvals)
