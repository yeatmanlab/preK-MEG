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
                           prep_cluster_stats)

# flags
tfce = True
n_jobs = 10

# config paths
data_root, subjects_dir, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'group-level', 'npz')
cluster_dir = os.path.join(results_dir, 'pskt', 'group-level', 'cluster')
for _dir in (cluster_dir,):
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
threshold = dict(start=0, step=0.1) if tfce else None
cluster_sigma = 0.

# load fsaverage source space to get connectivity matrix
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)
connectivity = mne.spatial_src_connectivity(fsaverage_src)

# load one STC to get bin centers
stc_path = os.path.join(results_dir, 'pskt', 'group-level', 'stc',
                        'GrandAvg-post_camp-pskt-5_sec-fft-avg-stc.h5')
stc = mne.read_source_estimate(stc_path)
all_freqs = stc.times
del stc


def find_clusters(X, fpath, onesamp=False, **kwargs):
    if onesamp:
        stat_fun = ttest_1samp_no_p
        cluster_fun = partial(permutation_cluster_1samp_test,
                              sigma=cluster_sigma)
    else:
        stat_fun = ttest_ind_no_p
        cluster_fun = partial(permutation_cluster_test, sigma=cluster_sigma)
    cluster_results = cluster_fun(X, stat_fun=stat_fun, **kwargs)
    stats = prep_cluster_stats(cluster_results)
    np.savez(fpath, **stats)


# load in all the data
data_npz = np.load(os.path.join(in_dir, 'data.npz'))
noise_npz = np.load(os.path.join(in_dir, 'noise.npz'))
# make mutable (NpzFile is not)
data_dict = {k: v for k, v in data_npz.items()}
noise_dict = {k: v for k, v in noise_npz.items()}
data_npz.close()
noise_npz.close()

# transpose because for clustering we need (subj, freq, space)
for s in subjects:
    for tpt in timepoints:
        data_dict[f'{s}-{tpt}'] = data_dict[f'{s}-{tpt}'].transpose(1, 0)
        noise_dict[f'{s}-{tpt}'] = noise_dict[f'{s}-{tpt}'].transpose(1, 0)

pre_data_dict = {k: v for k, v in data_dict.items() if k.endswith('pre')}
pre_noise_dict = {k: v for k, v in noise_dict.items() if k.endswith('pre')}
post_data_dict = {k: v for k, v in data_dict.items() if k.endswith('post')}
post_noise_dict = {k: v for k, v in noise_dict.items() if k.endswith('post')}

# confirmatory/reproduction: detectable effect in bins of interest in GrandAvg
pre_data = np.array(list(pre_data_dict.values()))
pre_noise = np.array(list(pre_noise_dict.values()))
post_data = np.array(list(post_data_dict.values()))
post_noise = np.array(list(post_noise_dict.values()))
pre_snr = pre_data - pre_noise
post_snr = post_data - post_noise

# planned comparison: pre-intervention median split on letter awareness test
upper_data = [data_dict[f'{s}-pre'] for s in groups['UpperKnowledge']]
lower_data = [data_dict[f'{s}-pre'] for s in groups['LowerKnowledge']]
upper_noise = [noise_dict[f'{s}-pre'] for s in groups['UpperKnowledge']]
lower_noise = [noise_dict[f'{s}-pre'] for s in groups['LowerKnowledge']]
upper_snr = np.array(upper_data) - np.array(upper_noise)
lower_snr = np.array(lower_data) - np.array(lower_noise)

# planned comparison: post-minus-pre-intervention, language-vs-letter cohort
lett_data = [data_dict[f'{s}-post'] - data_dict[f'{s}-pre'] for s in groups['LetterIntervention']]
lang_data = [data_dict[f'{s}-post'] - data_dict[f'{s}-pre'] for s in groups['LanguageIntervention']]
lett_noise = [noise_dict[f'{s}-post'] - noise_dict[f'{s}-pre'] for s in groups['LetterIntervention']]
lang_noise = [noise_dict[f'{s}-post'] - noise_dict[f'{s}-pre'] for s in groups['LanguageIntervention']]

lett_snr = np.array(lett_data) - np.array(lett_noise)
lang_snr = np.array(lang_data) - np.array(lang_noise)

del data_dict, noise_dict

# get the bin numbers we care about
these_freqs = (2, 4, 6, 12)
bin_idxs = {freq: np.argmin(np.abs(all_freqs - freq)) for freq in these_freqs}

for freq, bin_idx in bin_idxs.items():
    grandavg_pre_X = [pre_snr[:, [bin_idx], :]]
    grandavg_post_X = [post_snr[:, [bin_idx], :]]
    median_split_X = [upper_snr[:, [bin_idx], :], lower_snr[:, [bin_idx], :]]
    intervention_X = [lett_snr[:, [bin_idx], :], lang_snr[:, [bin_idx], :]]
    grandavg_pre_fname = 'GrandAvg-pre_camp'
    grandavg_post_fname = 'GrandAvg-post_camp'
    median_split_fname = 'UpperVsLowerKnowledge-pre_camp'
    intervention_fname = 'LetterVsLanguageIntervention-PostMinusPre_camp'
    for prefix, X in {grandavg_pre_fname: grandavg_pre_X,
                      grandavg_post_fname: grandavg_post_X,
                      median_split_fname: median_split_X,
                      intervention_fname: intervention_X}.items():
        fname = f'{prefix}-{freq}_Hz-SNR-clusters.npz'
        kwargs = dict(connectivity=connectivity, threshold=threshold,
                      n_permutations=1024, n_jobs=n_jobs, seed=rng,
                      buffer_size=1024, step_down_p=0.05, out_type='indices')
        # different kwargs for 1samp vs independent tests
        onesamp = prefix in (grandavg_pre_fname, grandavg_post_fname)
        find_clusters(X, os.path.join(cluster_dir, fname), onesamp, **kwargs)

# # cluster across all frequencies. Don't use regularization or step-down here
# # (need to save memory)
# if all_bins:
#     grandavg_X = [all_data]
#     median_split_X = [upper_data, lower_data]
#     intervention_X = [letter_data, language_data]
#     grandavg_fname = 'GrandAvg-PreAndPost_camp'
#     median_split_fname = 'LowerVsUpperKnowledge-pre_camp'
#     intervention_fname = 'LetterVsLanguageIntervention-PostMinusPre_camp'
#     for prefix, X in {grandavg_fname: grandavg_X,
#                       median_split_fname: median_split_X,
#                       intervention_fname: intervention_X}.items():
#         onesamp = prefix == grandavg_fname
#         stat_fun = partial(ttest_ind_no_p, sigma=cluster_sigma)
#         cluster_results = permutation_cluster_test(
#             X, connectivity=connectivity, threshold=threshold,
#             n_permutations=1024, n_jobs=n_jobs, seed=rng, buffer_size=1024,
#             stat_fun=stat_fun, step_down_p=0., out_type='indices')
#         stats = prep_cluster_stats(cluster_results)
#         fname = f'{prefix}-all_freqs-SNR-{hemi}-clusters.npz'
#         np.savez(os.path.join(cluster_dir, fname), **stats)
