#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import yaml
from functools import partial


def load_params(skip=True):
    """Load experiment parameters from YAML files."""
    paramdir = os.path.join('..', '..', 'params')
    yamload = partial(yaml.load, Loader=yaml.FullLoader)
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
