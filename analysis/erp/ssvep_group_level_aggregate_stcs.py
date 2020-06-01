#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Aggregate frequency-domain STCs across subjects.
"""

import os
import numpy as np
import mne
from aux_functions import (load_paths, load_params, load_cohorts,
                           div_by_adj_bins)

# config paths
_, _, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'stc', 'morphed-to-fsaverage')
stc_dir = os.path.join(results_dir, 'pskt', 'group-level', 'stc')
for _dir in (stc_dir,):
    os.makedirs(_dir, exist_ok=True)

# load params
_, _, subjects = load_params()
intervention_group, letter_knowledge_group = load_cohorts()
groups = dict(GrandAvg=subjects)
groups.update(intervention_group)
groups.update(letter_knowledge_group)

# config other
timepoints = ('pre', 'post')
trial_dur = 20
subdivide_epochs = 5
subdiv = f'-{subdivide_epochs}_sec' if subdivide_epochs else ''

# loop over timepoints
for timepoint in timepoints:
    # loop over cohort groups
    for group, members in groups.items():
        # only do pretest knowledge comparison for pre-camp timepoint
        if group.endswith('Knowledge') and timepoint == 'post':
            continue
        # aggregate over group members
        avg_data = 0.
        snr_data = 0.
        log_data = 0.
        for s in members:
            fname = f'{s}FSAverage-{timepoint}_camp-pskt{subdiv}-fft-stc.h5'
            stc = mne.read_source_estimate(os.path.join(in_dir, fname),
                                           subject='fsaverage')
            # convert complex values to magnitude
            this_data = np.abs(stc.data)
            avg_data += this_data
            # divide each bin by its neighbors on each side to get "SNR"
            this_snr = div_by_adj_bins(this_data)
            snr_data += this_snr
            log_data += np.log(this_snr)
        # save untransformed data & SNR data
        for kind, _data in zip(['avg', 'snr', 'log'],
                               [avg_data, snr_data, log_data]):
            # use a copy of the last STC as container
            this_stc = stc.copy()
            this_stc.data = _data / len(members)
            # save stc
            fname = f'{group}-{timepoint}_camp-pskt{subdiv}-fft-{kind}'
            this_stc.save(os.path.join(stc_dir, fname), ftype='h5')
