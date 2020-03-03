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

# load fsaverage source space
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)
fsaverage_src = mne.add_source_space_distances(fsaverage_src, dist_limit=0)

# define which labels comprise each region. Region 0 is a large label covering
# most of ventral occipitotemporal cortex. Regions 1-5 are sub-bands within
# region 0, starting with 1 at the occipital pole and moving anteriorly.
reg = {0: [f'lateraloccipital_{n}-lh' for n in (6, 7, 8, 9, 10, 11)] +
          [f'lingual_{n}-lh' for n in (1, 4, 8)] +
          [f'fusiform_{n}-lh' for n in (1, 2, 3)] +
          ['inferiortemporal_8-lh'],
       1: ['lingual_2-lh'] + [f'lateraloccipital_{n}-lh' for n in (4, 5)],
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
    merged_label = merged_label.restrict(fsaverage_src)
    merged_label = merged_label.fill(fsaverage_src)
    merged_label.save(f'ventral_ROI_{region_number}-lh.label')
