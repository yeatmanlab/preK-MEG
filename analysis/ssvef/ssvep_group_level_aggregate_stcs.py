#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Aggregate frequency-domain STCs across subjects.
"""

import os
import numpy as np
import mne
from sswef_helpers.aux_functions import (load_paths, load_params, load_cohorts,
                                         div_by_adj_bins)

# config paths
_, _, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'stc', 'morphed-to-fsaverage')
stc_dir = os.path.join(results_dir, 'pskt', 'group-level', 'stc')
for _dir in (stc_dir,):
    os.makedirs(_dir, exist_ok=True)

# load params
*_, subjects, cohort = load_params(experiment='pskt')
intervention_group, letter_knowledge_group = load_cohorts(experiment='pskt')
groups = dict(GrandAvg=subjects)
groups.update(intervention_group)
groups.update(letter_knowledge_group)

# inverse params
constraints = ('free',)  # 'loose', 'fixed')
estim_types = ('magnitude',)  # 'vector' 'normal')

# config other
timepoints = ('pre', 'post')
trial_dur = 20
conditions = ('ps', 'kt', 'all')

# loop over cortical estimate orientation constraints
for constr in constraints:
    print(f'{constr}')
    # loop over estimate types
    for estim_type in estim_types:
        print(f'  {estim_type}')
        if constr == 'fixed' and estim_type == 'normal':
            continue  # not implemented
        if constr == 'fixed' and estim_type == 'vector':
            continue  # not implemented
        # make the output directory if needed
        out_dir = f'{constr}-{estim_type}'
        os.makedirs(os.path.join(stc_dir, out_dir), exist_ok=True)
        # loop over timepoints
        for timepoint in timepoints:
            print(f'    {timepoint}')
            # loop over trial types
            for condition in conditions:
                # loop over cohort groups
                for group, members in groups.items():
                    # only do pretest knowledge comparison for pre-camp timept.
                    if group.endswith('Knowledge') and timepoint == 'post':
                        continue
                    # aggregate over group members
                    abs_data = 0.
                    snr_data = 0.
                    for s in members:
                        fname = (f'{s}FSAverage-{timepoint}_camp-pskt-'
                                 f'{condition}-fft-stc.h5')
                        fpath = os.path.join(in_dir, out_dir, fname)
                        stc = mne.read_source_estimate(
                            fpath, subject='fsaverage')
                        # convert complex values to magnitude
                        abs_data += np.abs(stc.data)
                        # divide each bin by neighbors to get "SNR"
                        snr_data += div_by_adj_bins(np.abs(stc.data))
                    # save untransformed data & SNR data
                    for kind, _data in zip(['amp', 'snr'],
                                           [abs_data, snr_data]):
                        # use a copy of the last STC as container
                        this_stc = stc.copy()
                        this_stc.data = _data / len(members)
                        # save stc
                        fname = (f'{cohort}-{group}-{timepoint}_camp-pskt'
                                 f'-{condition}-fft-{kind}')
                        fpath = os.path.join(stc_dir, out_dir, fname)
                        this_stc.save(fpath, ftype='h5')
