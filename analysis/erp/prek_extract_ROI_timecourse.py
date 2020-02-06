#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Extract mean activation time course from region-of-interest.
"""

import os
import re
import numpy as np
import pandas as pd
from mayavi import mlab
import mne
from aux_functions import load_paths, load_params, load_cohorts

mlab.options.offscreen = True
mne.cuda.init_cuda()
n_jobs = 10

# load params
brain_plot_kwargs, movie_kwargs, subjects = load_params()

# config paths
data_root, subjects_dir, results_dir = load_paths()
cluster_root = os.path.join(results_dir, 'clustering')
# get most current cluster subfolder
with open(os.path.join(cluster_root, 'most-recent-clustering.txt'), 'r') as f:
    cluster_dir = f.readline().strip()
ts_dir = os.path.join(cluster_dir, 'time-series')
os.makedirs(ts_dir, exist_ok=True)

# load cohort info (keys Language/LetterIntervention and Lower/UpperKnowledge)
intervention_group, letter_knowledge_group = load_cohorts()

# assemble groups to iterate over
groups_dict = dict(GrandAvg=subjects)
groups_dict.update(intervention_group)
groups_dict.update(letter_knowledge_group)

# load fsaverage source space
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)
hemi_nverts = len(fsaverage_src[0]['vertno'])


# helper function
def build_group_key(group):
    if group in ('letter', 'language'):
        key = f'{group.capitalize()}Intervention'
    elif group in ('upper', 'lower'):
        key = f'{group.capitalize()}Knowledge'
    elif group == 'grandavg':
        key = 'GrandAvg'
    return key


# workhorse function
def extract_time_course(cluster_fname):
    """."""
    # load the cluster results
    cluster_fpath = os.path.join(cluster_dir, cluster_fname)
    # DICT KEYS: clusters tvals pvals hzero good_cluster_idxs n_clusters
    cluster_dict = np.load(cluster_fpath, allow_pickle=True)
    signif_clu = cluster_dict['good_cluster_idxs'][0]

    # load the group-average STC
    avg_stc_fname = re.sub(r'(_[lr]h)?\.npz$', '', cluster_fname)
    avg_stc_fpath = os.path.join(results_dir, 'group_averages', avg_stc_fname)
    avg_stc = mne.read_source_estimate(avg_stc_fpath)

    # extract condition names from filenames
    group, timepoint, method, condition = avg_stc_fname.split('_')
    hemi_str = re.sub(r'\.npz$', '', cluster_fname)[-2:]

    # convert condition names as needed
    timepoints = dict(preCamp=['pre'], postCamp=['post'],
                      PostCampMinusPreCamp=['post', 'pre'])[timepoint]
    conditions = [x.lower() for x in condition.split('Minus')]
    groups = re.sub(r'(Intervention|Knowledge)$', '', group[:group.index('N')])
    groups = [x.lower() for x in groups.split('Minus')]

    # loop over clusters
    for clu in signif_clu:
        temporal_idxs, spatial_idxs = cluster_dict['clusters'][clu]
        # make sure we're right about which hemisphere the cluster is in
        if all(spatial_idxs <= hemi_nverts):
            assert hemi_str == 'lh'
            hemi = 0
        elif all(spatial_idxs >= hemi_nverts):
            assert hemi_str == 'rh'
            hemi = 1
            spatial_idxs = spatial_idxs - hemi_nverts
        else:
            err = ("you seem to have a cluster that spans "
                   "hemispheres, this shouldn't happen.")
            raise RuntimeError(err)
        # select the vertices in the cluster & convert to Label
        verts = np.unique(avg_stc.vertices[hemi][spatial_idxs])
        label = mne.Label(verts, hemi=hemi_str, subject='fsaverage')
        label = label.restrict(fsaverage_src)

        time_courses = dict()
        # loop over conditions to handle subtractions
        for timept in timepoints:
            time_courses[timept] = dict()
            for con in conditions:
                time_courses[timept][con] = dict()
                for grp in groups:
                    time_courses[timept][con][grp] = dict()
                    group_key = build_group_key(grp)
                    group_members = groups_dict[group_key]
                    for s in group_members:
                        this_subj = os.path.join(data_root, f'{timept}_camp',
                                                 'twa_hp', 'erp', s, 'stc')
                        fname = f'{s}FSAverage_{timept}Camp_{method}_{con}'
                        stc_path = os.path.join(this_subj, fname)
                        stc = mne.read_source_estimate(stc_path)
                        # extract label time course
                        kwargs = dict(src=fsaverage_src, mode='pca_flip',
                                      allow_empty=True)
                        time_course = mne.extract_label_time_course(stc, label,
                                                                    **kwargs)
                        time_courses[timept][con][grp][s] = np.squeeze(time_course)  # noqa E501
                    # convert dict of each subj's time series to DataFrame
                    df = pd.DataFrame(time_courses[timept][con][grp],
                                      index=stc.times)
                    time_courses[timept][con][grp] = df
                # collapse contrasts
                if len(groups) > 1:
                    time_courses[timept][con] = (time_courses[timept][con][groups[0]] -  # noqa E501
                                                 time_courses[timept][con][groups[1]])   # noqa E501
                else:
                    time_courses[timept][con] = time_courses[timept][con][grp]
            if len(conditions) > 1:
                time_courses[timept] = (time_courses[timept][conditions[0]] -
                                        time_courses[timept][conditions[1]])
            else:
                time_courses[timept] = time_courses[timept][con]
        if len(timepoints) > 1:
            time_courses = (time_courses[timepoints[0]] -
                            time_courses[timepoints[1]])
        else:
            time_courses = time_courses[timept]
        # save DataFrame
        ts_fname = f'{avg_stc_fname}_cluster{clu:05}.csv'
        time_courses.to_csv(os.path.join(ts_dir, ts_fname))


cluster_fnames = sorted([x.name for x in os.scandir(cluster_dir)
                         if x.is_file()])
for cluster_fname in cluster_fnames:
    extract_time_course(cluster_fname)
