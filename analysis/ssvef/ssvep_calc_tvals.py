#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Compute uncorrected t-value maps.
"""

import os
import numpy as np
from nibabel.freesurfer.io import write_morph_data
import mne
from mne.stats import ttest_ind_no_p, ttest_1samp_no_p
from analysis.aux_functions import (load_paths, load_params, load_cohorts,
                                    load_inverse_params)

# load params
brain_plot_kwargs, _, subjects, cohort = load_params(experiment='pskt')
inverse_params = load_inverse_params()
intervention_group, letter_knowledge_group = load_cohorts(experiment='pskt')
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
src_fname = os.path.join(subjects_dir, 'fsaverage', 'bem',
                         'fsaverage-ico-5-src.fif')
for _dir in (tval_dir,):
    os.makedirs(_dir, exist_ok=True)

# config other
timepoints = ('pre', 'post')
conditions = ('all', 'ps', 'kt')
sigma = 1e-3  # hat adjustment for low variance
src = mne.read_source_spaces(src_fname)
morph = mne.compute_source_morph(
    src, subjects_dir=subjects_dir, spacing=None, smooth='nearest')
freqs = mne.read_source_estimate(os.path.join(
    stc_dir, 'original-GrandAvg-pre_camp-pskt-all-fft-snr-stc.h5')).times
write_freqs = [2., 4., 6., 12.]
assert np.in1d(write_freqs, freqs).all()
offsets = np.cumsum([0, len(src[0]['rr']), len(src[1]['rr'])])
offsets = dict(lh=offsets[:-1],
               rh=offsets[1:])


def save_tvals(fname, tvals, freqs):
    """Save tvals along with freesurfer overlays."""
    assert tvals.shape == (20484, len(freqs))
    np.save(fname, tvals)
    for f, t in zip(freqs, tvals.T):
        if f in write_freqs:
            vals = morph.morph_mat @ t
            assert vals.shape == (327684,)
            for hemi, (start, stop) in offsets.items():
                use_fname = f'{os.path.splitext(fname)[0]}_{f:.0f}_{hemi}.curv'
                write_morph_data(use_fname, vals[start:stop])


for condition in conditions:
    print(f'Computing t-vals for {condition}')
    # load in all the data, making mutable (NpzFile is not)
    with np.load(os.path.join(npz_dir, f'snr-{condition}.npz')) as snr_npz:
        snr_dict = dict(snr_npz.items())
    with np.load(os.path.join(npz_dir, f'data-{condition}.npz')) as data_npz:
        data_dict = dict(data_npz.items())
    with np.load(os.path.join(npz_dir, f'noise-{condition}.npz')) as noise_npz:
        noise_dict = dict(noise_npz.items())

    # across-subj 1-sample t-vals (freq bin versus mean of 4 surrounding bins)
    for tpt in timepoints:
        snr_ = np.array([snr_dict[f'{s}-{tpt}'] for s in groups['GrandAvg']])
        data = np.array([data_dict[f'{s}-{tpt}'] for s in groups['GrandAvg']])
        nois = np.array([noise_dict[f'{s}-{tpt}'] for s in groups['GrandAvg']])
        assert snr_.dtype == np.float64
        assert data.dtype == np.float64
        assert nois.dtype == np.float64
        freq_ix = np.nonzero(freqs == 6.)[0]  # 6 Hz
        x = data[:, :, freq_ix] - nois[:, :, freq_ix]
        bad = np.where(np.percentile(x, 75, axis=-1) < 0)[0]
        if len(bad):
            bad = '\n'.join(groups['GrandAvg'][si] for si in bad)
            raise RuntimeError(
                'Look at subject(s) whose 75th percentile SNR is less than 0 '
                f'in {repr(tpt)}:\n{bad}')
        # here we use an ungainly list comprehension instead of:
        # tvals = ttest_1samp_no_p(data - nois, sigma=sigma)
        # because we want ttest_1samp_no_p to vectorize over vertices
        # but *not* over freq bins (the "hat" adjustment shouldn't be
        # taking other freq bins into account)
        tvals = np.array([ttest_1samp_no_p(_dmn, sigma=sigma) for _dmn in
                          (data - nois).transpose(2, 0, 1)]).T
        fname = f'DataMinusNoise1samp-{tpt}_camp-{condition}-tvals.npy'
        save_tvals(os.path.join(tval_dir, fname), tvals, freqs)
        # compute grand avg of SNR, and sanity check it
        ave = np.mean(snr_, axis=0)
        check_fname = os.path.join(
            stc_dir,
            f'original-GrandAvg-{tpt}_camp-pskt-{condition}'
            '-fft-snr-stc.h5')
        check_stc = mne.read_source_estimate(check_fname)
        np.testing.assert_allclose(ave, check_stc.data)
        fname = f'GrandAvg-{tpt}_camp-{condition}-grandavg.npy'
        save_tvals(os.path.join(tval_dir, fname), ave, freqs)
        if condition == 'all' and tpt == 'pre':
            print(check_fname)
            print(os.path.join(tval_dir, fname))

    # planned comparison: group split on pre-intervention letter awareness test
    # 2-sample t-test on differences between SNRs
    median_split = list()
    for group in ('UpperKnowledge', 'LowerKnowledge'):
        snr = np.array([snr_dict[f'{s}-pre'] for s in groups[group]])
        assert snr.dtype == np.float64
        median_split.append(snr)
    median_split_tvals = np.array([
        ttest_ind_no_p(a, b, sigma=sigma) for a, b in zip(
            median_split[0].transpose(2, 0, 1),
            median_split[1].transpose(2, 0, 1))]).T

    # planned comparison: post-minus-pre-intervention, language-vs-letter group
    # 2-sample t-test on differences between SNRs
    intervention = list()
    if cohort == 'replication':
        print('Skipping t-test for intervention group for replication cohort.')
        intervention_tvals = np.array([])
    else:
        for group in ('LetterIntervention', 'LanguageIntervention'):
            snr = np.array([snr_dict[f'{s}-post'] - snr_dict[f'{s}-pre']
                            for s in groups[group]])
            assert snr.dtype == np.float64
            intervention.append(snr)
        intervention_tvals = np.array([
            ttest_ind_no_p(a, b, sigma=sigma) for a, b in zip(
                intervention[0].transpose(2, 0, 1),
                intervention[1].transpose(2, 0, 1))]).T

    # save the data
    tval_dict = {'UpperVsLowerKnowledge-pre_camp': median_split_tvals,
                 'LetterVsLanguageIntervention-PostMinusPre_camp': intervention_tvals}  # noqa E501
    for fname, tvals in tval_dict.items():
        save_tvals(os.path.join(tval_dir, f'{fname}-{condition}-tvals.npy'),
                   tvals, freqs)
