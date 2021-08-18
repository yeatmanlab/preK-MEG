#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Run stats & clustering on SSVEP data.
"""

from functools import partial
import os
import time
import numpy as np
import mne
from scipy import stats
from mne.stats import (permutation_cluster_test, ttest_ind_no_p,
                       permutation_cluster_1samp_test, ttest_1samp_no_p)
from analysis.aux_functions import (load_paths, load_params, load_cohorts,
                                    prep_cluster_stats, load_inverse_params)
ppf = stats.t.ppf
del stats

# flags
tfce = True
n_jobs = 10

# load params
*_, subjects, cohort = load_params(experiment='pskt')
inverse_params = load_inverse_params()
intervention_group, letter_knowledge_group = load_cohorts()
groups = dict(GrandAvg=subjects)
groups.update(intervention_group)
groups.update(letter_knowledge_group)

# config paths
data_root, subjects_dir, results_dir = load_paths()
chosen_constraints = ('{orientation_constraint}-{estimate_type}'
                      ).format_map(inverse_params)

tval_dir = os.path.join(results_dir, 'pskt', 'group-level', 'tvals',
                        chosen_constraints)
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
conditions = ('all', 'ps', 'kt')
rng = np.random.RandomState(seed=15485863)  # the one millionth prime
cluster_sigma = 0.001

# load fsaverage source space to get connectivity matrix
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)
adjacency = mne.spatial_src_adjacency(fsaverage_src)

# load one STC to get bin centers
file_prefix = 'all' if cohort == 'pooled' else cohort
stc_path = os.path.join(
    results_dir, 'pskt', 'group-level', 'stc', chosen_constraints,
    f'{file_prefix}-GrandAvg-post_camp-pskt-all-fft-amp-stc.h5')
stc = mne.read_source_estimate(stc_path)
all_freqs = stc.times
del stc


def find_clusters(X, fpath, qc_tvals, onesamp=False, **kwargs):
    if onesamp:
        stat_fun = ttest_1samp_no_p
        cluster_fun = permutation_cluster_1samp_test
        df = len(X) - 1
    else:
        stat_fun = ttest_ind_no_p
        cluster_fun = permutation_cluster_test
        df = len(X[0]) + len(X[1]) - 2
    # threshold=None is only valid for f_oneway in permutation_cluster_test,
    # and even for the onesamp variant it warns because we use partial, so
    # let's just explicitly set it:
    if kwargs['threshold'] is None:
        kwargs['threshold'] = -ppf(0.05 / 2., df)

    stat_fun = partial(stat_fun, sigma=cluster_sigma)
    # sanity check: stat_fun tvals vs manually-computed tvals
    stat_fun_X = [X] if onesamp else X
    np.testing.assert_array_equal(qc_tvals, stat_fun(*stat_fun_X))
    cluster_results = cluster_fun(X, stat_fun=stat_fun, **kwargs)
    # sanity check: cluster tvals vs manually-computed tvals
    # (only valid if not using TFCE)
    if not tfce:
        np.testing.assert_allclose(qc_tvals, cluster_results[0])
    stats = prep_cluster_stats(cluster_results)
    stats['tfce'] = tfce
    stats['threshold'] = kwargs['threshold']
    np.savez(fpath, **stats)


t0 = time.time()
for condition in conditions:
    print(f'Running {condition}')
    # load in all the data
    snr_npz = np.load(os.path.join(npz_dir, f'snr-{condition}.npz'))
    data_npz = np.load(os.path.join(npz_dir, f'data-{condition}.npz'))
    noise_npz = np.load(os.path.join(npz_dir, f'noise-{condition}.npz'))
    # make mutable (NpzFile is not)
    snr_dict = {k: v for k, v in snr_npz.items()}
    data_dict = {k: v for k, v in data_npz.items()}
    noise_dict = {k: v for k, v in noise_npz.items()}
    snr_npz.close()
    data_npz.close()
    noise_npz.close()

    # confirmatory/reproduction: detectable effect in GrandAvg bins of interest
    pre_dict = {k: data_dict[k] - noise_dict[k] for k in data_dict
                if k.endswith('pre')}
    post_dict = {k: data_dict[k] - noise_dict[k] for k in data_dict
                 if k.endswith('post')}
    pre_data = np.array(list(pre_dict.values()))
    post_data = np.array(list(post_dict.values()))

    # planned comparison: group split on pre-intervention letter awareness test
    upper_data = np.array([snr_dict[f'{s}-pre'] for s in groups['UpperKnowledge']])  # noqa E501
    lower_data = np.array([snr_dict[f'{s}-pre'] for s in groups['LowerKnowledge']])  # noqa E501

    # planned comparison: post-minus-pre-intervention, language-vs-letter group
    lett_data = np.array([snr_dict[f'{s}-post'] - snr_dict[f'{s}-pre']
                          for s in groups['LetterIntervention']])
    lang_data = np.array([snr_dict[f'{s}-post'] - snr_dict[f'{s}-pre']
                          for s in groups['LanguageIntervention']])

    del snr_dict, data_dict, noise_dict, pre_dict, post_dict

    # get the bin numbers we care about
    these_freqs = (2, 4, 6, 12)
    bin_idxs = {freq: np.argmin(np.abs(all_freqs - freq))
                for freq in these_freqs}

    for freq, bin_idx in bin_idxs.items():
        grandavg_pre_X = pre_data[..., bin_idx]
        grandavg_post_X = post_data[..., bin_idx]
        median_split_X = [upper_data[..., bin_idx],
                          lower_data[..., bin_idx]]
        intervention_X = [lett_data[..., bin_idx],
                          lang_data[..., bin_idx]]
        grandavg_pre_fname = 'DataMinusNoise1samp-pre_camp'
        grandavg_post_fname = 'DataMinusNoise1samp-post_camp'
        median_split_fname = 'UpperVsLowerKnowledge-pre_camp'
        intervention_fname = 'LetterVsLanguageIntervention-PostMinusPre_camp'
        for prefix, X in {grandavg_pre_fname: grandavg_pre_X,
                          grandavg_post_fname: grandavg_post_X,
                          median_split_fname: median_split_X,
                          intervention_fname: intervention_X}.items():
            fname = f'{prefix}-{condition}-{freq}_Hz-SNR-clusters.npz'
            # kwargs for clustering function
            onesamp = prefix in (grandavg_pre_fname, grandavg_post_fname)
            func = ttest_1samp_no_p if onesamp else (lambda x: ttest_ind_no_p(*x))  # noqa E501
            # include at most 25% of the brain
            # start, top = np.percentile(np.abs(func(X)), [75, 99])
            # step = min(start, top / 10)
            # threshold = dict(start=start, step=step) if tfce else None
            threshold = dict(start=0, step=0.2) if tfce else None
            kwargs = dict(adjacency=adjacency, threshold=threshold,
                          n_permutations=10000, n_jobs=n_jobs, seed=rng,
                          buffer_size=None,
                          step_down_p=0.05,
                          out_type='indices', verbose=True)
            tval_fname = f'{prefix}-{condition}-tvals.npy'
            qc_tvals = np.load(os.path.join(tval_dir, tval_fname))[:, bin_idx]
            find_clusters(X, os.path.join(cluster_dir, fname), qc_tvals,
                          onesamp, **kwargs)
print(f'Completed in {time.time() - t0:0.1f} seconds')
