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


def define_labels(region, action, hemi):
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
                                    subjects_dir=None,
                                    regexp=regexp)
    n_hemis = 2 if hemi == 'both' else 1
    assert len(labels) == n_hemis * len(label_names)
    # merge the labels using the sum(..., start) hack
    merged_label = sum(labels[1:], labels[0])
    assert len(merged_label.name.split('+')) == n_hemis * len(label_names)
    return merged_label
