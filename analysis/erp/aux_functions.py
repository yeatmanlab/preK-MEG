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
