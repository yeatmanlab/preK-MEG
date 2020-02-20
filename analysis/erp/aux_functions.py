#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import yaml
from functools import partial
import numpy as np

paramdir = os.path.join('..', '..', 'params')
yamload = partial(yaml.load, Loader=yaml.FullLoader)


def load_params(skip=True):
    """Load experiment parameters from YAML files."""
    with open(os.path.join(paramdir, 'brain_plot_params.yaml'), 'r') as f:
        brain_plot_kwargs = yamload(f)
    with open(os.path.join(paramdir, 'movie_params.yaml'), 'r') as f:
        movie_kwargs = yamload(f)
    with open(os.path.join(paramdir, 'subjects.yaml'), 'r') as f:
        subjects = yamload(f)
    if skip:
        with open(os.path.join(paramdir, 'skip_subjects.yaml'), 'r') as f:
            skips = yamload(f)
        subjects = sorted(set(subjects) - set(skips))
    return brain_plot_kwargs, movie_kwargs, subjects


def load_cohorts():
    """load intervention and knowledge groups."""
    with open(os.path.join(paramdir, 'intervention_cohorts.yaml'), 'r') as f:
        intervention_group = yamload(f)
    with open(os.path.join(paramdir, 'letter_knowledge_cohorts.yaml'),
              'r') as f:
        letter_knowledge_group = yamload(f)
    # convert 1103 → 'prek_1103'
    for _dict in (intervention_group, letter_knowledge_group):
        for key, value in _dict.items():
            _dict[key] = [f'prek_{num}' for num in value]
    return intervention_group, letter_knowledge_group


def load_paths():
    """Load necessary filesystem paths."""
    with open(os.path.join(paramdir, 'paths.yaml'), 'r') as f:
        paths = yamload(f)
    return paths['data_root'], paths['subjects_dir'], paths['results_dir']


def prep_cluster_stats(cluster_results):
    (tvals, clusters, cluster_pvals, hzero) = cluster_results
    stats = dict(n_clusters=len(clusters),
                 clusters=clusters,
                 tvals=tvals,
                 pvals=cluster_pvals,
                 # this is multicomparison-corrected already:
                 good_cluster_idxs=np.where(cluster_pvals < 0.05),
                 hzero=hzero)
    return stats


def define_labels(region, action, hemi, subjects_dir=None):
    from mne import read_labels_from_annot
    if action not in ('include', 'exclude'):
        raise ValueError('action must be "include" or "exclude".')

    region_dict = {'medial-wall': ('G_and_S_paracentral',      # 3
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
                                   ),
                   'VOTC': ('G_and_S_occipital_inf',      # 2
                            'G_oc-temp_lat-fusifor',      # 21
                            'G_temporal_inf',             # 37
                            'S_collat_transv_ant',        # 50
                            'S_collat_transv_post',       # 51
                            'S_occipital_ant',            # 59
                            'S_oc-temp_lat',              # 60
                            'S_oc-temp_med_and_Lingual',  # 61
                            'S_temporal_inf',             # 72
                            )
                   }

    valid_regions = tuple(region_dict)
    if region not in valid_regions + (None,):
        raise ValueError(f'region must be "{", ".join(valid_regions)}", or '
                         'None.')

    label_names = region_dict[region]
    regexp = r'|'.join(label_names)
    # DON'T DO IT THIS WAY (PARCELLATION NOT GUARANTEED TO BE EXHAUSTIVE):
    # if action == 'include':
    #     # select for exclusion all labels *except* those given
    #     regexp = f'(?!{regexp})'
    labels = read_labels_from_annot(subject='fsaverage',
                                    parc='aparc.a2009s',
                                    hemi=hemi,
                                    subjects_dir=subjects_dir,
                                    regexp=regexp)
    n_hemis = 2 if hemi == 'both' else 1
    assert len(labels) == n_hemis * len(label_names)
    # merge the labels using the sum(..., start) hack
    merged_label = sum(labels[1:], labels[0])
    assert len(merged_label.name.split('+')) == n_hemis * len(label_names)
    return merged_label


def get_stc_from_conditions(method, timepoint, condition, subject):
    """Load an STC file for the given experimental conditions.

    Parameters
    ----------

    method : 'dSPM' | 'sLORETA'

    timepoint : 'pre' | 'post'

    condition : 'words' | 'faces' | 'cars' | 'aliens'

    subject : str
        Can be a subject identifier ('prek_1103'), a group name ('language',
        'letter', 'upper', or 'lower'), or `None` (to get grand average of all
        subjects).
    """
    from mne import read_source_estimate
    data_root, _, results_dir = load_paths()
    # allow both groups and subjects as the "subject" argument
    group_map = {None: 'GrandAvg',
                 'language': 'LanguageInterventionN24',
                 'letter': 'LetterInterventionN24',
                 'upper': 'UpperKnowledgeN24',
                 'lower': 'LowerKnowledgeN24'}
    # if "subject" is not in group_map, use it as-is
    subject = group_map.get(subject, subject)
    # filename pattern
    fname = f'{subject}FSAverage_{timepoint}Camp_{method}_{condition}'
    if subject in group_map.values():
        folder = os.path.join(results_dir, 'group_averages')
    else:
        folder = os.path.join(data_root, f'{timepoint}_camp', 'twa_hp', 'erp',
                              subject, 'stc')
    stc_path = os.path.join(folder, fname)
    stc = read_source_estimate(stc_path)
    return stc


def get_dataframe_from_label(label, src, methods=('dSPM', 'sLORETA'),
                             timepoints=('pre', 'post'),
                             conditions=('words', 'faces', 'cars', 'aliens'),
                             subjects=None):
    """Get average timecourse within label across all subjects."""
    from pandas import DataFrame, concat, melt
    from mne import Label, extract_label_time_course
    # allow looping over single Label
    if isinstance(label, Label):
        label = [label]
    # load subjects list
    if subjects is None:
        _, _, subjects = load_params()
    elif isinstance(subjects, str):
        subjects = [subjects]
    # load cohort information
    intervention_group, letter_knowledge_group = load_cohorts()
    intervention_map = {subj: group.lower()[:-12]
                        for group, members in intervention_group.items()
                        for subj in members}
    knowledge_map = {subj: group.lower()[:-9]
                     for group, members in letter_knowledge_group.items()
                     for subj in members}

    time_courses = dict()
    # loop over source localization algorithms
    for method in methods:
        time_courses[method] = dict()
        # loop over pre/post measurement time
        for timept in timepoints:
            time_courses[method][timept] = dict()
            # loop over conditions
            for cond in conditions:
                time_courses[method][timept][cond] = dict()
                # loop over subjects
                for subj in subjects:
                    # load STC
                    stc = get_stc_from_conditions(method, timept, cond, subj)
                    # extract label time course
                    time_courses[method][timept][cond][subj] = np.squeeze(
                        extract_label_time_course(stc, label, src=src,
                                                  mode='pca_flip'))
                # convert dict of each subj's time series to DataFrame
                df = DataFrame(time_courses[method][timept][cond],
                               index=range(len(stc.times)))
                df['time'] = stc.times
                time_courses[method][timept][cond] = df
                # store current "condition" in a column before exiting loop
                time_courses[method][timept][cond]['condition'] = cond
            # combine DataFrames across conditions
            dfs = (time_courses[method][timept][c] for c in conditions)
            time_courses[method][timept] = concat(dfs)
            # store current "timepoint" in a column before exiting loop
            time_courses[method][timept]['timepoint'] = timept
        # combine DataFrames across timepoints
        dfs = (time_courses[method][t] for t in timepoints)
        time_courses[method] = concat(dfs)
        # store current "method" in a column before exiting loop
        time_courses[method]['method'] = method
    # combine DataFrames across methods
    dfs = (time_courses[m] for m in methods)
    time_courses = concat(dfs)

    # reshape DataFrame
    all_cols = time_courses.columns.values
    subj_cols = time_courses.columns.str.startswith('prek')
    id_vars = all_cols[np.logical_not(subj_cols)]
    time_courses = melt(time_courses, id_vars=id_vars, var_name='subj')
    # add columns for intervention cohort and pretest letter knowledge
    time_courses['intervention'] = time_courses['subj'].map(intervention_map)
    time_courses['pretest'] = time_courses['subj'].map(knowledge_map)
    return time_courses
