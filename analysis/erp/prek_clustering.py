#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Perform spatiotemporal clustering on STCs.

Treat each subject as an observation of shape n_times × n_vertices, and stack
all observations for a given pair of conditions (e.g., "faces" vs. "words") to
perform parametric spatiotemporal clustering.
"""

import os
from functools import partial
from itertools import combinations
import numpy as np
import mne
from mne.stats import spatio_temporal_cluster_1samp_test
from aux_functions import (load_paths, load_params, load_cohorts,
                           prep_cluster_stats)

mne.cuda.init_cuda()
rng = np.random.RandomState(seed=15485863)  # the one millionth prime
n_jobs = 10
threshold = None        # or dict(start=0, step=0.2) for TFCE
spatial_exclude = True  # None -> whole brain; True -> use labels listed below


def subset_X(X, hemi, conn_matrices, exclusion):
    """Subset the cluster matrix, connectivity & exclusion label by hemi."""
    this_X = (X[..., :hemi_nverts] if hemi == 'lh' else X[..., hemi_nverts:])
    this_conn = conn_matrices[hemi]
    this_excl = exclusion[hemi].vertices
    results = dict(X=this_X, connectivity=this_conn, spatial_exclude=this_excl)
    return results


def cluster_each_hemi(X, conn_matrices, exclusion, group, timepoint, method,
                      con):
    for hemi in ('lh', 'rh'):
        cluster_input = subset_X(X, hemi, conn_matrices, exclusion)
        cluster_results = one_samp_test(**cluster_input)
        stats = prep_cluster_stats(cluster_results)
        # save clustering results
        out_fname = f'{group}_{timepoint}_{method}_{con}_{hemi}.npz'
        out_fpath = os.path.join(cluster_dir, out_fname)
        np.savez(out_fpath, **stats)


# load params
_, _, subjects = load_params()

# config paths
data_root, subjects_dir, results_dir = load_paths()
groupavg_path = os.path.join(results_dir, 'group_averages')

# set cache dir
cache_dir = os.path.join(data_root, 'cache')
os.makedirs(cache_dir, exist_ok=True)
mne.set_cache_dir(cache_dir)

# config other
methods = ('dSPM', 'sLORETA')  # dSPM, sLORETA, eLORETA
timepoints = ('preCamp', 'postCamp')
conditions = ['words', 'faces', 'cars']  # we purposely omit 'aliens' here
contrasts = {f'{contr[0].capitalize()}Minus{contr[1].capitalize()}': contr
             for contr in list(combinations(conditions, 2))}

# load cohort info (keys Language/LetterIntervention and Lower/UpperKnowledge)
intervention_group, letter_knowledge_group = load_cohorts()

# assemble groups to iterate over
groups = dict(GrandAvg=subjects)
groups.update(intervention_group)
groups.update(letter_knowledge_group)

# load fsaverage source space to get connectivity matrix
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)
hemi_nverts = len(fsaverage_src[0]['vertno'])

# make separate source spaces for each hemisphere
lh_src = fsaverage_src.copy()
rh_src = fsaverage_src.copy()
_ = lh_src.pop(1)
_ = rh_src.pop(0)

conn_matrices = dict(both=mne.spatial_src_connectivity(fsaverage_src),
                     lh=mne.spatial_src_connectivity(lh_src),
                     rh=mne.spatial_src_connectivity(rh_src))

if spatial_exclude is not None:
    # labels to exclude, see 10.1016/j.neuroimage.2010.06.010
    label_descr = 'medial-wall'
    label_names = ('G_and_S_paracentral',      # 3
                   'G_and_S_cingul-Ant',       # 6
                   'G_and_S_cingul-Mid-Ant',   # 7
                   'G_and_S_cingul-Mid-Post',  # 8
                   'G_cingul-Post-dorsal',     # 9
                   'G_cingul-Post-ventral',    # 10
                   'G_front_sup',              # 16
                   'G_oc-temp_med-Parahip',    # 23
                   'G_precuneus',              # 30
                   'G_rectus',                 # 31
                   'G_subcallosal',            # 32
                   'S_cingul-Marginalis',      # 46
                   'S_pericallosal',           # 66
                   'S_suborbital',             # 70
                   'S_subparietal',            # 71
                   'Unknown',
                   )
    regexp = '|'.join(label_names)
    exclusion = dict()
    for hemi_idx, hemi in enumerate(('lh', 'rh')):
        exclusion[hemi] = mne.read_labels_from_annot(
            subject='fsaverage', parc='aparc.a2009s', hemi=hemi,
            subjects_dir=subjects_dir, regexp=regexp)
        assert len(exclusion[hemi]) == len(label_names)
        # merge the labels using the sum(..., start) hack
        exclusion[hemi] = sum(exclusion[hemi][1:], exclusion[hemi][0])
        assert len(exclusion[hemi].name.split('+')) == len(label_names)
        exclusion[hemi].vertices = \
            np.intersect1d(exclusion[hemi].vertices,
                           fsaverage_src[hemi_idx]['vertno'])
        if hemi == 'rh':
            exclusion[hemi].vertices += hemi_nverts
    name = ' + '.join(exclusion[hemi].name for hemi in ('lh', 'rh'))
    exclusion['both'] = mne.BiHemiLabel(exclusion['lh'], exclusion['rh'],
                                        name=name)
    exclusion['both'].vertices = np.concatenate([exclusion['lh'].vertices,
                                                 exclusion['rh'].vertices])
# cluster results get different subfolders depending on threshold / exclude
cluster_root = os.path.join(results_dir, 'clustering')
if spatial_exclude is not None:
    cluster_subdir = f"exclude-{label_descr}"
else:
    cluster_subdir = 'whole-brain'
cluster_subsubdir = '.'
if isinstance(threshold, dict):
    cluster_subsubdir = 'tfce_{start}_{step}'.format_map(threshold)
elif threshold is not None:
    cluster_subsubdir = f'thresh_{threshold}'
# write most recently used cluster dir to file
cluster_dir = os.path.join(cluster_root, cluster_subdir, cluster_subsubdir)
with open(os.path.join(cluster_root, 'most-recent-clustering.txt'), 'w') as f:
    f.write(cluster_dir)
os.makedirs(cluster_dir, exist_ok=True)

# prepare clustering function
one_samp_test = partial(spatio_temporal_cluster_1samp_test,
                        threshold=threshold,
                        n_permutations=1024,
                        n_jobs=n_jobs,
                        seed=rng,
                        buffer_size=1024)

# loop over source localization algorithms
for method in methods:
    data_dict = dict()
    # loop over groups
    for group_name, group_members in groups.items():
        group = f'{group_name}N{len(group_members)}FSAverage'
        data_dict[group] = dict()
        # loop over pre/post measurement time
        for timepoint in timepoints:
            # skip conditions we don't need / care about
            if group_name.endswith('Knowledge') and timepoint == 'postCamp':
                continue
            data_dict[group][timepoint] = dict()
            # load the individual subject STCs for each condition
            for cond in conditions:
                data_dict[group][timepoint][cond] = list()
                # loop over subjects
                for s in group_members:
                    this_subj = os.path.join(data_root,
                                             f'{timepoint[:-4]}_camp',
                                             'twa_hp', 'erp', s, 'stc')
                    fname = f'{s}FSAverage_{timepoint}_{method}_{cond}'
                    stc_path = os.path.join(this_subj, fname)
                    stc = mne.read_source_estimate(stc_path)
                    stc_data = stc.data.transpose(1, 0)  # noqa; need (subj, time, space)
                    data_dict[group][timepoint][cond].append(stc_data)
                data_dict[group][timepoint][cond] = \
                    np.array(data_dict[group][timepoint][cond])

            # CONTRAST TRIAL CONDITIONS
            for con, (contr_0, contr_1) in contrasts.items():
                X = (data_dict[group][timepoint][contr_0] -
                     data_dict[group][timepoint][contr_1])
                data_dict[group][timepoint][con] = X
                cluster_each_hemi(X, conn_matrices, exclusion, group,
                                  timepoint, method, con)

        # CONTRAST POST-MINUS-PRE
        timepoint = 'PostCampMinusPreCamp'
        data_dict[group][timepoint] = dict()
        for con in conditions + list(contrasts):
            # skip conditions we don't need / care about
            if group_name.endswith('Knowledge'):
                continue
            X = (data_dict[group]['postCamp'][con] -
                 data_dict[group]['preCamp'][con])
            data_dict[group][timepoint][con] = X
            cluster_each_hemi(X, conn_matrices, exclusion, group,
                              timepoint, method, con)

    # CONTRAST PRE-INTERVENTION LETTER KNOWLEDGE
    timepoint = 'preCamp'
    group_name = 'UpperMinusLowerKnowledge'
    n_subj = {g: len(groups[g]) for g in letter_knowledge_group}
    n = '-'.join([str(n_subj[g]) for g in letter_knowledge_group])
    group = f'{group_name}N{n}FSAverage'
    data_dict[group] = dict()
    data_dict[group][timepoint] = dict()
    keys = {g: f'{g}N{n_subj[g]}FSAverage' for g in letter_knowledge_group}
    for con in conditions + list(contrasts):
        X = (data_dict[keys['UpperKnowledge']][timepoint][con] -
             data_dict[keys['LowerKnowledge']][timepoint][con])
        data_dict[group][timepoint][con] = X
        cluster_each_hemi(X, conn_matrices, exclusion, group,
                          timepoint, method, con)

    # CONTRAST EFFECT OF INTERVENTION ON COHORTS
    # # skip this one because it doesn't make sense to do the subtraction when
    # # dealing with individual subject data instead of averages.
    # timepoint = 'PostCampMinusPreCamp'
    # group_name = 'LetterMinusLanguageIntervention'
    # n_subj = {g: len(groups[g]) for g in intervention_group}
    # n = '-'.join([str(n_subj[g]) for g in intervention_group])
    # group = f'{group_name}N{n}FSAverage'
    # data_dict[group] = dict()
    # data_dict[group][timepoint] = dict()
    # keys = {g: f'{g}N{n_subj[g]}FSAverage' for g in intervention_group}
    # for con in conditions + list(contrasts):
    #     X = (data_dict[keys['LetterIntervention']][timepoint][con] -
    #          data_dict[keys['LanguageIntervention']][timepoint][con])
    #     data_dict[group][timepoint][con] = X
    #     cluster_each_hemi(X, conn_matrices, exclusion, group,
    #                       timepoint, method, con)
