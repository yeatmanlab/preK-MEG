#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Make uncorrected t-value maps.
"""

import os
import numpy as np
import mne
from mne.stats import ttest_ind_no_p, ttest_1samp_no_p
from aux_functions import (load_paths, load_params, load_cohorts,
                           div_by_adj_bins)

# flags
hemi = 'lh'

# config paths
data_root, subjects_dir, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'stc', 'morphed-to-fsaverage')
stc_dir = os.path.join(results_dir, 'pskt', 'group-level', 'stc')
fig_dir = os.path.join(results_dir, 'pskt', 'fig', 'tvals')
tval_dir = os.path.join(results_dir, 'pskt', 'group-level', 'tvals')
for _dir in (tval_dir, fig_dir):
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
for s in groups['GrandAvg']:
    for timepoint in timepoints:
        stub = f'{s}FSAverage-{timepoint}_camp-pskt{subdiv}-fft'
        stc = mne.read_source_estimate(os.path.join(in_dir, f'{stub}-stc.h5'),
                                       subject='fsaverage')
        # convert to SNR across bins & save for later group comparisons
        data_dict[f'{s}-{timepoint}'] = div_by_adj_bins(np.abs(stc.data))

# across-subj 1-sample t-values
for tpt in timepoints:
    data = np.array([data_dict[f'{s}-{tpt}'] for s in groups['GrandAvg']])
    tvals = ttest_1samp_no_p(data)
    np.save(os.path.join(tval_dir, f'GrandAvg-{tpt}_camp-tvals.npy'), tvals)

# planned comparison: pre-intervention median split on letter awareness test
med_spl = [np.array([data_dict[f'{s}-pre'] for s in groups['UpperKnowledge']]),
           np.array([data_dict[f'{s}-pre'] for s in groups['LowerKnowledge']])]

# planned comparison: post-minus-pre-intervention, language-vs-letter cohort
interv = [np.array([data_dict[f'{s}-post'] - data_dict[f'{s}-pre']
                    for s in groups['LetterIntervention']]),
          np.array([data_dict[f'{s}-post'] - data_dict[f'{s}-pre']
                    for s in groups['LanguageIntervention']])]

med_spl_tvals = ttest_ind_no_p(*med_spl)
interv_tvals = ttest_ind_no_p(*interv)

# save the data
tval_dict = {'LowerVsUpperKnowledge-pre_camp': med_spl_tvals,
             'LetterVsLanguageIntervention-PostMinusPre_camp': interv_tvals}
for fname, tvals in tval_dict.items():
    np.save(os.path.join(tval_dir, fname), tvals)
