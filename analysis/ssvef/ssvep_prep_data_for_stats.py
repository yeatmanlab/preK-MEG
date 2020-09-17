#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

aggregate signal and noise data into .npz files.
"""

import os
import numpy as np
import mne
from analysis.aux_functions import (load_paths, load_params, div_by_adj_bins,
                                    load_inverse_params)

# load params
brain_plot_kwargs, _, subjects, cohort = load_params()
inverse_params = load_inverse_params()

# config paths
data_root, subjects_dir, results_dir = load_paths()
chosen_constraints = ('{orientation_constraint}-{estimate_type}'
                      ).format_map(inverse_params)

in_dir = os.path.join(results_dir, 'pskt', 'stc', 'morphed-to-fsaverage',
                      chosen_constraints)
npz_dir = os.path.join(results_dir, 'pskt', 'group-level', 'npz',
                       chosen_constraints)
for _dir in (npz_dir,):
    os.makedirs(_dir, exist_ok=True)

# config other
timepoints = ('pre', 'post')
subdivide_epochs = 5
subdiv = f'-{subdivide_epochs}_sec' if subdivide_epochs else ''

# load in all the data
data_dict = dict()
noise_dict = dict()
for s in subjects:
    print('Working on subject %s.' % s)
    for timepoint in timepoints:
        stub = f'{s}FSAverage-{timepoint}_camp-pskt{subdiv}-fft'
        stc = mne.read_source_estimate(os.path.join(in_dir, f'{stub}-stc.h5'),
                                       subject='fsaverage')
        # compute magnitude (signal) & avg of adjacent bins on either side
        # (noise), & save for later group comparisons
        data_dict[f'{s}-{timepoint}'] = np.abs(stc.data)
        noise_dict[f'{s}-{timepoint}'] = div_by_adj_bins(np.abs(stc.data),
                                                         return_noise=True)
np.savez(os.path.join(npz_dir, 'data.npz'), **data_dict)
np.savez(os.path.join(npz_dir, 'noise.npz'), **noise_dict)
