#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Compute uncorrected t-value maps.
"""

import os
import numpy as np
from mne.stats import ttest_ind_no_p, ttest_1samp_no_p
from analysis.aux_functions import (load_paths, load_params, load_cohorts,
                                    load_inverse_params)

# load params
brain_plot_kwargs, _, subjects, cohort = load_params()
inverse_params = load_inverse_params()
intervention_group, letter_knowledge_group = load_cohorts()
groups = dict(GrandAvg=subjects)
groups.update(intervention_group)
groups.update(letter_knowledge_group)

# config paths
data_root, subjects_dir, results_dir = load_paths()
chosen_constraints = ('{orientation_constraint}-{estimate_type}'
                      ).format_map(inverse_params)

npz_dir = os.path.join(results_dir, 'pskt', 'group-level', 'npz',
                       chosen_constraints)
tval_dir = os.path.join(results_dir, 'pskt', 'group-level', 'tvals',
                        chosen_constraints)
for _dir in (tval_dir,):
    os.makedirs(_dir, exist_ok=True)

# config other
timepoints = ('pre', 'post')

# load in all the data
data_npz = np.load(os.path.join(npz_dir, 'data.npz'))
noise_npz = np.load(os.path.join(npz_dir, 'noise.npz'))
# make mutable (NpzFile is not)
data_dict = {k: v for k, v in data_npz.items()}
noise_dict = {k: v for k, v in noise_npz.items()}
data_npz.close()
noise_npz.close()
print('Data dict keys: %s' % data_dict.keys())

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
if cohort == 'replication':
    print('Skipping t-values for intervention group for replication cohort.')
    intervention_tvals = []
else:
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
