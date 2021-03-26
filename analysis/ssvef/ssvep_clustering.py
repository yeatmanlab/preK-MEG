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
from analysis.aux_functions import (load_paths, load_params, load_cohorts,
                                    prep_cluster_stats, load_inverse_params)

# flags
tfce = True
n_jobs = 10

# load params
*_, subjects, cohort = load_params()
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
cluster_dir = os.path.join(results_dir, 'pskt', 'group-level', 'cluster',
                           chosen_constraints)
for _dir in (cluster_dir,):
    os.makedirs(_dir, exist_ok=True)

# set cache dir
cache_dir = os.path.join(data_root, 'cache')
os.makedirs(cache_dir, exist_ok=True)
mne.set_cache_dir(cache_dir)

# config other
timepoints = ('pre', 'post')
conditions = ('ps', 'kt', 'all')
subdivide_epochs = 5
subdiv = f'-{subdivide_epochs}_sec' if subdivide_epochs else ''
rng = np.random.RandomState(seed=15485863)  # the one millionth prime
threshold = dict(start=0, step=0.1) if tfce else None
cluster_sigma = 0.

# load fsaverage source space to get connectivity matrix
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)
connectivity = mne.spatial_src_connectivity(fsaverage_src)

# load one STC to get bin centers
file_prefix = 'all' if cohort == 'pooled' else cohort
stc_path = os.path.join(
    results_dir, 'pskt', 'group-level', 'stc', chosen_constraints,
    f'{file_prefix}-GrandAvg-post_camp-pskt-5_sec-all-fft-amp-stc.h5')
stc = mne.read_source_estimate(stc_path)
all_freqs = stc.times
del stc


def find_clusters(X, fpath, onesamp=False, **kwargs):
    if onesamp:
        stat_fun = ttest_1samp_no_p
        cluster_fun = permutation_cluster_1samp_test
    else:
        stat_fun = ttest_ind_no_p
        cluster_fun = permutation_cluster_test
    stat_fun = partial(stat_fun, sigma=cluster_sigma)
    cluster_results = cluster_fun(X, stat_fun=stat_fun, **kwargs)
    stats = prep_cluster_stats(cluster_results)
    np.savez(fpath, **stats)


for condition in conditions:
    # load in all the data
    data_npz = np.load(os.path.join(npz_dir, f'data-{condition}.npz'))
    noise_npz = np.load(os.path.join(npz_dir, f'noise-{condition}.npz'))
    # make mutable (NpzFile is not) and transpose because for clustering we
    # need (subj, freq, space)
    data_dict = {k: v.transpose(1, 0) for k, v in data_npz.items()}
    noise_dict = {k: v.transpose(1, 0) for k, v in noise_npz.items()}
    data_npz.close()
    noise_npz.close()

    pre_data_dict = {k: v for k, v in data_dict.items() if k.endswith('pre')}
    pre_noise_dict = {k: v for k, v in noise_dict.items() if k.endswith('pre')}
    post_data_dict = {k: v for k, v in data_dict.items() if k.endswith('post')}
    post_noise_dict = {k: v for k, v in noise_dict.items() if k.endswith('post')}  # noqa E501

    # confirmatory/reproduction: detectable effect in GrandAvg bins of interest
    pre_data = np.array(list(pre_data_dict.values()))
    pre_noise = np.array(list(pre_noise_dict.values()))
    post_data = np.array(list(post_data_dict.values()))
    post_noise = np.array(list(post_noise_dict.values()))
    pre_snr = pre_data - pre_noise
    post_snr = post_data - post_noise

    # planned comparison: group split on pre-intervention letter awareness test
    upper_data = np.array([data_dict[f'{s}-pre'] for s in groups['UpperKnowledge']])  # noqa E501
    lower_data = np.array([data_dict[f'{s}-pre'] for s in groups['LowerKnowledge']])  # noqa E501

    # planned comparison: post-minus-pre-intervention, language-vs-letter group
    lett_data = np.array([data_dict[f'{s}-post'] - data_dict[f'{s}-pre']
                          for s in groups['LetterIntervention']])
    lang_data = np.array([data_dict[f'{s}-post'] - data_dict[f'{s}-pre']
                          for s in groups['LanguageIntervention']])

    del data_dict, noise_dict

    # get the bin numbers we care about
    these_freqs = (2, 4, 6, 12)
    bin_idxs = {freq: np.argmin(np.abs(all_freqs - freq))
                for freq in these_freqs}

    for freq, bin_idx in bin_idxs.items():
        grandavg_pre_X = pre_snr[:, [bin_idx], :]
        grandavg_post_X = post_snr[:, [bin_idx], :]
        median_split_X = [upper_data[:, [bin_idx], :],
                          lower_data[:, [bin_idx], :]]
        intervention_X = [lett_data[:, [bin_idx], :],
                          lang_data[:, [bin_idx], :]]
        grandavg_pre_fname = 'GrandAvg-pre_camp'
        grandavg_post_fname = 'GrandAvg-post_camp'
        median_split_fname = 'UpperVsLowerKnowledge-pre_camp'
        intervention_fname = 'LetterVsLanguageIntervention-PostMinusPre_camp'
        for prefix, X in {grandavg_pre_fname: grandavg_pre_X,
                          grandavg_post_fname: grandavg_post_X,
                          median_split_fname: median_split_X,
                          intervention_fname: intervention_X}.items():
            fname = f'{prefix}-{condition}-{freq}_Hz-SNR-clusters.npz'
            # kwargs for clustering function
            kwargs = dict(connectivity=connectivity, threshold=threshold,
                          n_permutations=1024, n_jobs=n_jobs, seed=rng,
                          buffer_size=1024, step_down_p=0.05,
                          out_type='indices')
            onesamp = prefix in (grandavg_pre_fname, grandavg_post_fname)
            find_clusters(X, os.path.join(cluster_dir, fname), onesamp,
                          **kwargs)
