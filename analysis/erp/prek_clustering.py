#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Perform spatiotemporal clustering on STCs.

Treat each subject as an observation of shape n_times × n_vertices, and stack
all observations for a given pair of conditions (e.g., "faces" vs. "words") to
perform parametric spatiotemporal clustering.
"""

import os
from itertools import combinations
import numpy as np
import mne
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       spatio_temporal_cluster_test)
from aux_functions import (load_paths, load_params, load_cohorts,
                           prep_cluster_stats, define_labels)

mne.cuda.init_cuda()
rng = np.random.RandomState(seed=15485863)  # the one millionth prime
n_jobs = 10
threshold = None        # or dict(start=0, step=0.2) for TFCE


def do_clustering(X, label, connectivity, groups=1):
    fun = (spatio_temporal_cluster_1samp_test if groups == 1 else
           spatio_temporal_cluster_test)
    cluster_results = fun(X, spatial_exclude=label.vertices,
                          connectivity=connectivity,
                          threshold=threshold, n_permutations=1024,
                          n_jobs=n_jobs, seed=rng, buffer_size=1024)
    stats = prep_cluster_stats(cluster_results)
    # save clustering results
    out_fname = f'{group}_{timepoint}_{method}_{con}_{hemi}.npz'  # noqa
    out_fpath = os.path.join(cluster_dir, out_fname)
    np.savez(out_fpath, **stats)


# define the maximum spatial extent of clustering. The following special
# values are defined:
#
# dict(action='exclude', region=None)          → whole brain
# dict(action='exclude', region='medial-wall') → exclude medial wall
# dict(action='include', region='VOTC')        → exclude all except that region
#
# See the define_labels() function for region definitions. Hemi should be a
# list of strings, containing one or more of ("lh", "rh", "both").
spatial_limits = dict(action='include', region='VOTC', hemi=['lh'])

# load params
_, _, subjects = load_params()

# config paths
data_root, subjects_dir, results_dir = load_paths()

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
# make separate source spaces and connectivity matrices for each hemisphere
lh_src = fsaverage_src.copy()
rh_src = fsaverage_src.copy()
_ = lh_src.pop(1)
_ = rh_src.pop(0)
source_spaces = dict(both=fsaverage_src, lh=lh_src, rh=rh_src)
conn_matrices = {hemi: mne.spatial_src_connectivity(src)
                 for hemi, src in source_spaces.items()}

for hemi in spatial_limits['hemi']:
    hemi_idx = dict(lh=0, rh=1, both=(0, 1))[hemi]
    source_space = source_spaces[hemi]
    conn_matrix = conn_matrices[hemi]
    # get the label
    label = define_labels(region=spatial_limits['region'],
                          action=spatial_limits['action'],
                          hemi=hemi, subjects_dir=subjects_dir)
    # make sure label vertex density matches source space density
    label.restrict(source_space)
    # if label is region to *include*, get complement of vertices to use as
    # *exclusion* set
    if spatial_limits['action'] == 'include':
        if hemi == 'both':
            label.lh.vertices = np.setdiff1d(source_space[0]['vertno'],
                                             label.lh.vertices)
            label.rh.vertices = np.setdiff1d(source_space[1]['vertno'],
                                             label.rh.vertices)
        else:
            # use [0] here regardless of hemi because we have separate
            # single-hemi source spaces
            label.vertices = np.setdiff1d(source_space[0]['vertno'],
                                          label.vertices)

    # cluster results get different subfolders depending on spatial exclude...
    cluster_root = os.path.join(results_dir, 'clustering')
    if spatial_limits['region'] is None:
        cluster_subdir = f'whole-brain-{hemi}'
    else:
        cluster_subdir = ('{action}-{region}-'.format_map(spatial_limits) +
                          f'{hemi}')
    # ...and different sub-subfolders depending on thresholding method
    cluster_subsubdir = '.'
    if isinstance(threshold, dict):
        cluster_subsubdir = 'tfce_{start}_{step}'.format_map(threshold)
    elif threshold is not None:
        cluster_subsubdir = f'thresh_{threshold}'
    # write most recently used cluster dir to file
    cluster_dir = os.path.join(cluster_root, cluster_subdir, cluster_subsubdir)
    cluster_path = os.path.join(cluster_root, 'most-recent-clustering.txt')
    with open(cluster_path, 'w') as f:
        f.write(cluster_dir)
    os.makedirs(cluster_dir, exist_ok=True)

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
                if group_name.endswith('Knowledge') and \
                        timepoint == 'postCamp':
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
                        # get just the hemisphere(s) we want; transpose because
                        # we ultimately need (subj, time, space)
                        attr = dict(lh='lh_data', rh='rh_data', both='data')
                        stc_data = getattr(stc, attr[hemi]).transpose(1, 0)
                        data_dict[group][timepoint][cond].append(stc_data)
                    data_dict[group][timepoint][cond] = \
                        np.array(data_dict[group][timepoint][cond])

                # CONTRAST TRIAL CONDITIONS
                for con, (contr_0, contr_1) in contrasts.items():
                    X = (data_dict[group][timepoint][contr_0] -
                         data_dict[group][timepoint][contr_1])
                    data_dict[group][timepoint][con] = X
                    do_clustering(X, label, conn_matrix)

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
                do_clustering(X, label, conn_matrix)

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
            do_clustering(X, label, conn_matrix)

        # CONTRAST EFFECT OF INTERVENTION ON COHORTS
        # this uses a different stat function, and takes a list of arrays for X
        # instead of doing a subtraction (because subtraction would not have
        # been within-subject)
        timepoint = 'PostCampMinusPreCamp'
        group_name = 'LetterMinusLanguageIntervention'
        n_subj = {g: len(groups[g]) for g in intervention_group}
        n = '-'.join([str(n_subj[g]) for g in intervention_group])
        group = f'{group_name}N{n}FSAverage'
        data_dict[group] = dict()
        data_dict[group][timepoint] = dict()
        keys = {g: f'{g}N{n_subj[g]}FSAverage' for g in intervention_group}
        for con in conditions + list(contrasts):
            X = [data_dict[keys['LetterIntervention']][timepoint][con],
                 data_dict[keys['LanguageIntervention']][timepoint][con]]
            data_dict[group][timepoint][con] = X
            do_clustering(X, label, conn_matrix, groups=2)
