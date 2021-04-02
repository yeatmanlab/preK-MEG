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
stc_dir = os.path.join(results_dir, 'pskt', 'group-level', 'stc',
                       chosen_constraints)
tval_dir = os.path.join(results_dir, 'pskt', 'group-level', 'tvals',
                        chosen_constraints)
for _dir in (tval_dir,):
    os.makedirs(_dir, exist_ok=True)

# config other
timepoints = ('post', 'pre', 'post')
conditions = ('all', 'ps', 'kt')
sigma = 1e-3  # hat adjustment for low variance

for condition in conditions:
    print(f'Computing t-vals for {condition}')
    # load in all the data
    data_npz = np.load(os.path.join(npz_dir, f'data-{condition}.npz'))
    noise_npz = np.load(os.path.join(npz_dir, f'noise-{condition}.npz'))
    # make mutable (NpzFile is not)
    data_dict = {k: v for k, v in data_npz.items()}
    noise_dict = {k: v for k, v in noise_npz.items()}
    data_npz.close()
    noise_npz.close()

    # across-subj 1-sample t-vals (freq bin versus mean of 4 surrounding bins)
    for tpt in timepoints:
        data = np.array([data_dict[f'{s}-{tpt}'] for s in groups['GrandAvg']])
        nois = np.array([noise_dict[f'{s}-{tpt}'] for s in groups['GrandAvg']])
        assert data.dtype == np.float64
        assert nois.dtype == np.float64
        x = data[:, :, 30] - nois[:, :, 30]  # 6 Hz
        bad = np.where(np.percentile(x, 75, axis=-1) < 0)[0]
        if len(bad):
            bad = '\n'.join(groups['GrandAvg'][si] for si in bad)
            raise RuntimeError(
                'Look at subject(s) whose 75th percentile SNR is less than 0 '
                f'in {repr(tpt)}:\n{bad}')
        # TODO look at ps post
        tvals = np.array([ttest_1samp_no_p(d - n, sigma=sigma) for d, n in zip(
            data.transpose(2, 0, 1), nois.transpose(2, 0, 1))]).T
        fname = f'DataMinusNoise1samp-{tpt}_camp-{condition}-tvals.npy'
        np.save(os.path.join(tval_dir, fname), tvals)
        # compute SNR, save as GrandAvg, and sanity check it
        ave = np.mean(data / nois, axis=0)
        check_fname = os.path.join(
            stc_dir,
            f'original-GrandAvg-{tpt}_camp-pskt-5_sec-{condition}'
            '-fft-snr-stc.h5')
        check_stc = mne.read_source_estimate(check_fname)
        np.testing.assert_allclose(ave, check_stc.data)
        fname = f'GrandAvg-{tpt}_camp-{condition}-tvals.npy'
        np.save(os.path.join(tval_dir, fname), ave)
        if condition == 'all' and tpt == 'pre':
            print(check_fname)
            print(os.path.join(tval_dir, fname))

    # planned comparison: group split on pre-intervention letter awareness test
    # 2-sample t-test on SNRs
    median_split = list()
    for group in ('UpperKnowledge', 'LowerKnowledge'):
        data = np.array([data_dict[f'{s}-pre'] / noise_dict[f'{s}-pre']
                         for s in groups[group]])
        assert data.dtype == np.float64
        median_split.append(data)
    median_split_tvals = np.array([
        ttest_ind_no_p(a, b, sigma=sigma) for a, b in zip(
            median_split[0].transpose(2, 0, 1),
            median_split[1].transpose(2, 0, 1))]).T

    # planned comparison: post-minus-pre-intervention, language-vs-letter group
    # 2-sample on differences in SNRs
    intervention = list()
    if cohort == 'replication':
        print('Skipping t-test for intervention group for replication cohort.')
        intervention_tvals = np.array([])
    else:
        for group in ('LetterIntervention', 'LanguageIntervention'):
            data = np.array([data_dict[f'{s}-post'] / noise_dict[f'{s}-post'] -
                             data_dict[f'{s}-pre'] / noise_dict[f'{s}-pre']
                            for s in groups[group]])
            assert data.dtype == np.float64
            intervention.append(data)
        intervention_tvals = np.array([
            ttest_ind_no_p(a, b, sigma=sigma) for a, b in zip(
                intervention[0].transpose(2, 0, 1),
                intervention[1].transpose(2, 0, 1))]).T

    # save the data
    tval_dict = {'UpperVsLowerKnowledge-pre_camp': median_split_tvals,
                 'LetterVsLanguageIntervention-PostMinusPre_camp': intervention_tvals}  # noqa E501
    for fname, tvals in tval_dict.items():
        np.save(os.path.join(tval_dir, f'{fname}-{condition}-tvals.npy'),
                tvals)
