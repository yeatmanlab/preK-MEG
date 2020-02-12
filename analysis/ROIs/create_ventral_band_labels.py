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

# define which labels comprise each region
reg = {1: ['lingual_2-lh'] + [f'lateraloccipital_{n}-lh' for n in (4, 5)],
       2: ['lingual_1-lh'] + [f'lateraloccipital_{n}-lh' for n in (3, 6, 11)],
       3: ['lingual_4-lh', 'fusiform_2-lh'] + [f'lateraloccipital_{n}-lh'
                                               for n in (7, 10)],
       4: ['lingual_8-lh', 'lateraloccipital_9-lh', 'inferiortemporal_8-lh'] +
          [f'fusiform_{n}-lh' for n in (1, 3)],
       5: [f'fusiform_{n}-lh' for n in (4, 5)] +
          [f'inferiortemporal_{n}-lh' for n in (5, 6, 7)],
       }

label_kwargs = dict(subject='fsaverage', parc='aparc_sub', hemi='lh',
                    subjects_dir=subjects_dir)

for region_number, label_names in reg.items():
    regexp = r'|'.join(label_names)
    labels = mne.read_labels_from_annot(regexp=regexp, **label_kwargs)
    merged_label = sum(labels[1:], labels[0])
    merged_label.comment = ''
    merged_label.save(f'ventral_band_{region_number}-lh.label')
