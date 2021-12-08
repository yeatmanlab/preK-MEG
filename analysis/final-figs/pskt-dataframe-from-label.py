#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Prepare a dataframe from a given label.
"""

import os
import mne
from sswef_helpers.aux_functions import (load_paths, get_dataframe_from_label,
                                    load_fsaverage_src)

# config paths
data_root, subjects_dir, results_dir = load_paths()
roi_dir = os.path.join('..', 'ROIs')

# config other
methods = ('snr', 'fft')
conditions = ('ps', 'kt')
timepoints = ('post', 'pre')

# load label
fnames = ('MPM_IOS_IOG_lh.label', 'MPM_pOTS_lh.label')
labels = [mne.read_label(os.path.join(roi_dir, fname)) for fname in fnames]
label = labels[0] + labels[1]  # combine the two MPM labels
region = 'IOS_IOG_pOTS'

# load fsaverage source space
fsaverage_src = load_fsaverage_src()

# load brain data & restrict to label
df = get_dataframe_from_label(label, fsaverage_src, methods=methods,
                              conditions=conditions, unit='freq',
                              experiment='pskt')
df['roi'] = region

df.to_csv(f'pskt-in-label-{region}.csv', index=False)
