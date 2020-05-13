#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Run stats & clustering on SSVEP data.
"""

import os
from functools import partial
import numpy as np
import mne
from mne.stats import (permutation_cluster_test, ttest_ind_no_p,
                       permutation_cluster_1samp_test, ttest_1samp_no_p)
from aux_functions import (load_paths, load_params, load_cohorts,
                           div_by_adj_bins, prep_cluster_stats)

# flags
all_bins = False  # whether to run clustering across all bins or just (2, 4, 6)
run_clustering = False
tfce = True
n_jobs = 10
hemi = 'lh'

# config paths
data_root, subjects_dir, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'stc', 'morphed-to-fsaverage')
tval_dir = os.path.join(results_dir, 'pskt', 'group-level', 'tvals')
cluster_dir = os.path.join(results_dir, 'pskt', 'group-level', 'cluster')
for _dir in (tval_dir, cluster_dir):
    os.makedirs(_dir, exist_ok=True)

# set cache dir
cache_dir = os.path.join(data_root, 'cache')
os.makedirs(cache_dir, exist_ok=True)
mne.set_cache_dir(cache_dir)

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
threshold = dict(start=0, step=0.2) if tfce else None
stat_fun = partial(ttest_ind_no_p, equal_var=False)

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


def find_clusters(X, fpath, onesamp=False):
    cluster_func = (permutation_cluster_1samp_test if onesamp else
                    permutation_cluster_test)
    cluster_results = cluster_func(
        X, connectivity=connectivity, threshold=threshold,
        n_permutations=1024, n_jobs=n_jobs, seed=rng, buffer_size=1024,
        stat_fun=stat_fun, step_down_p=0.05, out_type='indices')
    stats = prep_cluster_stats(cluster_results)
    np.savez(fpath, **stats)


# load in all the data
data_dict = dict()
for s in groups['GrandAvg']:
    for timepoint in timepoints:
        fname = f'{s}FSAverage-{timepoint}_camp-pskt{subdiv}-fft-stc.h5'
        stc = mne.read_source_estimate(os.path.join(in_dir, fname),
                                       subject='fsaverage')
        # get just the hemisphere(s) we want
        data_attr = dict(lh='lh_data', rh='rh_data', both='data')
        # transpose because we ultimately need (subj, freq, space)
        stc_data = getattr(stc, data_attr[hemi]).transpose(1, 0)
        data_dict[f'{s}-{timepoint}'] = stc_data

# confirmatory/reproduction: detectable effect in bins of interest in GrandAvg
all_data = div_by_adj_bins(np.abs(list(data_dict.values())))

# planned comparison: pre-intervention median split on letter awareness test
lower_data = list()
upper_data = list()
for s in groups['LowerKnowledge']:
    lower_data.append(data_dict[f'{s}-pre'])
for s in groups['UpperKnowledge']:
    upper_data.append(data_dict[f'{s}-pre'])
lower_data = div_by_adj_bins(np.abs(lower_data))
upper_data = div_by_adj_bins(np.abs(upper_data))

# planned comparison: post-minus-pre-intervention, language-vs-letter cohort
letter_data = list()
language_data = list()
for s in groups['LetterIntervention']:
    letter_data.append(data_dict[f'{s}-post'] - data_dict[f'{s}-pre'])
for s in groups['LanguageIntervention']:
    language_data.append(data_dict[f'{s}-post'] - data_dict[f'{s}-pre'])
letter_data = div_by_adj_bins(np.abs(letter_data))
language_data = div_by_adj_bins(np.abs(language_data))

# get the bin numbers we care about
bin_idxs = dict()
for freq in (2, 4, 6):
    bin_idxs[freq] = np.argmin(np.abs(stc.times - freq))

del data_dict

for freq, bin_idx in bin_idxs.items():
    grandavg_X = [all_data[:, [bin_idx], :]]
    median_split_X = [upper_data[:, [bin_idx], :],
                      lower_data[:, [bin_idx], :]]
    intervention_X = [letter_data[:, [bin_idx], :],
                      language_data[:, [bin_idx], :]]
    grandavg_fname = 'GrandAvg-PreAndPost_camp'
    median_split_fname = 'LowerVsUpperKnowledge-pre_camp'
    intervention_fname = 'LetterVsLanguageIntervention-PostMinusPre_camp'
    for prefix, X in {grandavg_fname: grandavg_X,
                      median_split_fname: median_split_X,
                      intervention_fname: intervention_X}.items():
        onesamp = prefix == grandavg_fname
        # uncorrected t-maps
        t_func = (ttest_1samp_no_p if onesamp else
                  partial(ttest_ind_no_p, equal_var=False))
        fname = f'{prefix}-{freq}_Hz-SNR-{hemi}-tvals.npy'
        tvals = t_func(*X, sigma=1e-3)
        np.save(os.path.join(tval_dir, fname), tvals)
        # clustering
        if run_clustering:
            fname = f'{prefix}-{freq}_Hz-SNR-{hemi}-clusters.npz'
            find_clusters(X, os.path.join(cluster_dir, fname), onesamp)

# cluster across all frequencies. Don't use regularization or step-down here
# (need to save memory)
if all_bins:
    median_split_X = [upper_data, lower_data]
    intervention_X = [letter_data, language_data]
    median_split_fname = f'LowerVsUpperKnowledge-pre_camp'
    intervention_fname = f'LetterVsLanguageIntervention-PostMinusPre_camp'
    for prefix, X in {median_split_fname: median_split_X,
                      intervention_fname: intervention_X}.items():
        # uncorrected t-maps
        fname = f'{prefix}-all_freqs-SNR-{hemi}-tvals.npy'
        tvals = ttest_ind_no_p(*median_split_X, equal_var=False, sigma=1e-3)
        np.save(os.path.join(tval_dir, fname), tvals)
        if run_clustering:
            cluster_results = permutation_cluster_test(
                X, connectivity=connectivity, threshold=threshold,
                n_permutations=1024, n_jobs=n_jobs, seed=rng, buffer_size=1024,
                stat_fun=partial(ttest_ind_no_p, equal_var=False, sigma=0.),
                step_down_p=0., out_type='indices')
            stats = prep_cluster_stats(cluster_results)
            fname = f'{prefix}-all_freqs-SNR-{hemi}-clusters.npz'
            np.savez(os.path.join(cluster_dir, fname), **stats)
