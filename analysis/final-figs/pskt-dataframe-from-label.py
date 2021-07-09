#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Prepare a dataframe from a given label.
"""

import os
import mne
from analysis.aux_functions import load_paths, get_dataframe_from_label

# config paths
data_root, subjects_dir, results_dir = load_paths()
roi_dir = os.path.join('..', 'ROIs')

# config other
methods = ('snr', 'fft')
conditions = ('ps', 'kt')
timepoints = ('post', 'pre')

# load label
fname = '2Hz_LetterKnowledge.lh.label'
fpath = os.path.join(roi_dir, fname)
region = '2_Hz-LetterKnowledge'
label = mne.read_label(fpath)
# "that label was made based on the peak differences between High vs Low letter
# knowledge correcting for multiple comparisons based on FDR"


# load fsaverage source space
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)
fsaverage_src = mne.add_source_space_distances(fsaverage_src, dist_limit=0)

# load brain data & restrict to label
df = get_dataframe_from_label(label, fsaverage_src, methods=methods,
                              conditions=conditions, unit='freq')
df['roi'] = region

df.to_csv('pskt-in-label.csv', index=False)
