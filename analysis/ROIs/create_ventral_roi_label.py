#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Create ROI labels (adjacent bands along ventral surface of left hemisphere).
"""

import os
import yaml
import mne

# get subjects_dir
with open(os.path.join('..', '..', 'params', 'paths.yaml'), 'r') as f:
    subjects_dir = yaml.load(f, Loader=yaml.FullLoader)['subjects_dir']

# make sure we have the parcellation
mne.datasets.fetch_aparc_sub_parcellation(subjects_dir)

# define which labels comprise the region
label_names = ([f'lateraloccipital_{n}-lh' for n in (6, 7, 8, 9, 10, 11)] +
               [f'lingual_{n}-lh' for n in (1, 4, 8)] +
               [f'fusiform_{n}-lh' for n in (1, 2, 3)] +
               ['inferiortemporal_8-lh'])

label_kwargs = dict(subject='fsaverage', parc='aparc_sub', hemi='lh',
                    subjects_dir=subjects_dir)

regexp = r'|'.join(label_names)
labels = mne.read_labels_from_annot(regexp=regexp, **label_kwargs)
merged_label = sum(labels[1:], labels[0])
merged_label.comment = ''
merged_label.save(f'ventral_ROI-lh.label')
