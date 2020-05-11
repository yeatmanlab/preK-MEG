#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Run clustering on SSVEP data.
"""

import os
import numpy as np
import mne
from mne.stats import permutation_cluster_test
from aux_functions import (load_paths, load_params, load_cohorts,
                           div_by_adj_bins, prep_cluster_stats)

# config paths
_, subjects_dir, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'stc', 'morphed-to-fsaverage')
cluster_dir = os.path.join(results_dir, 'pskt', 'group-level', 'cluster')
os.makedirs(cluster_dir, exist_ok=True)

# load params
_, _, subjects = load_params()
intervention_group, letter_knowledge_group = load_cohorts()
groups = dict(GrandAvg=subjects)
groups.update(intervention_group)
groups.update(letter_knowledge_group)

# config other
timepoints = ('pre', 'post')
subdivide_epochs = 5
subdiv = f'-{subdivide_epochs}_sec' if subdivide_epochs else ''
rng = np.random.RandomState(seed=15485863)  # the one millionth prime
threshold = dict(start=0, step=0.2)  # or None to skip TFCE
n_jobs = 10
hemi = 'lh'

# load fsaverage source space to get connectivity matrix
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)
# make source space and connectivity matrix hemisphere-specific
if hemi == 'lh':
    _ = fsaverage_src.pop(1)
elif hemi == 'rh':
    _ = fsaverage_src.pop(0)
connectivity = mne.spatial_src_connectivity(fsaverage_src)


def do_clustering(X, fpath):
    cluster_results = permutation_cluster_test(
        X, connectivity=connectivity, threshold=threshold,
        n_permutations=1024, n_jobs=n_jobs, seed=rng, buffer_size=1024)
    stats = prep_cluster_stats(cluster_results)
    np.savez(fpath, **stats)


# load in all the data
all_data = dict()
for s in groups['GrandAvg']:
    for timepoint in timepoints:
        fname = f'{s}FSAverage-{timepoint}_camp-pskt{subdiv}-fft-stc.h5'
        stc = mne.read_source_estimate(os.path.join(in_dir, fname),
                                       subject='fsaverage')
        # get just the hemisphere(s) we want
        data_attr = dict(lh='lh_data', rh='rh_data', both='data')
        # transpose because we ultimately need (subj, freq, space)
        stc_data = getattr(stc, data_attr[hemi]).transpose(1, 0)
        all_data[f'{s}-{timepoint}'] = stc_data

# planned comparison: pre-intervention median split on letter awareness test
lower_data = list()
upper_data = list()
for s in groups['LowerKnowledge']:
    lower_data.append(all_data[f'{s}-pre'])
for s in groups['UpperKnowledge']:
    upper_data.append(all_data[f'{s}-pre'])
lower_data = div_by_adj_bins(np.abs(np.array(lower_data)))
upper_data = div_by_adj_bins(np.abs(np.array(upper_data)))

# planned comparison: post-minus-pre-intervention, language-vs-letter cohort
letter_data = list()
language_data = list()
for s in groups['LetterIntervention']:
    letter_data.append(all_data[f'{s}-post'] - all_data[f'{s}-pre'])
for s in groups['LanguageIntervention']:
    language_data.append(all_data[f'{s}-post'] - all_data[f'{s}-pre'])
letter_data = div_by_adj_bins(np.abs(np.array(letter_data)))
language_data = div_by_adj_bins(np.abs(np.array(language_data)))

# get the bin numbers we care about
bin_idxs = dict()
for freq in (2, 4, 6):
    bin_idxs[freq] = np.argmin(np.abs(stc.times - freq))

del all_data

for freq, bin_idx in bin_idxs.items():
    median_split_X = [lower_data[:, [bin_idx], :],
                      upper_data[:, [bin_idx], :]]
    intervention_X = [letter_data[:, [bin_idx], :],
                      language_data[:, [bin_idx], :]]
    median_split_fname = f'LowerVsUpperKnowledge-pre_camp-{freq}_Hz-SNR-{hemi}.npz'  # noqa E501
    intervention_fname = f'LetterVsLanguageIntervention-PostMinusPre_camp-{freq}_Hz-SNR-{hemi}.npz'  # noqa E501
    for fname, X in {median_split_fname: median_split_X,
                     intervention_fname: intervention_X}.items():
        do_clustering(X, os.path.join(cluster_dir, fname))
