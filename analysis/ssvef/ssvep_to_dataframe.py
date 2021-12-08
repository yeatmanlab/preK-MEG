#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Convert frequency-domain STCs to dataframes.
"""

import os
import numpy as np
import pandas as pd
import mne
from sswef_helpers.aux_functions import load_paths, load_params, load_inverse_params

# load params
_, _, subjects, cohort = load_params(experiment='pskt')
inverse_params = load_inverse_params()

# config paths
_, _, results_dir = load_paths()
chosen_constraints = ('{orientation_constraint}-{estimate_type}'
                      ).format_map(inverse_params)

in_dir = os.path.join(results_dir, 'pskt', 'stc', 'morphed-to-fsaverage',
                      chosen_constraints)
out_dir = os.path.join(results_dir, 'pskt', 'group-level', 'dataframe')
for _dir in (out_dir,):
    os.makedirs(_dir, exist_ok=True)

# config other
timepoints = ('pre', 'post')
conditions = ('ps', 'kt', 'all')

# loop over trial types
for condition in conditions:
    # container
    df = pd.DataFrame()
    # loop over timepoints
    for timepoint in timepoints:
        # loop over cohort groups
        for s in subjects:
            fname = (f'{s}FSAverage-{timepoint}_camp-pskt-'
                     f'{condition}-fft-stc.h5')
            stc = mne.read_source_estimate(os.path.join(in_dir, fname),
                                           subject='fsaverage')
            # convert complex values to magnitude
            stc.data = np.abs(stc.data)
            # convert to dataframe
            this_df = stc.to_data_frame(time_format=None, long_format=True)
            this_df.rename(columns=dict(time='freq'), inplace=True)
            this_df['subject'] = s
            this_df['timepoint'] = timepoint
            this_df['condition'] = condition
            # aggregate
            df = pd.concat((df, this_df), axis=0)

    fname = (f'all_subjects-fsaverage-{condition}-{chosen_constraints}-'
             'freq_domain-stc.csv')
    df.to_csv(os.path.join(out_dir, fname), index=False)
