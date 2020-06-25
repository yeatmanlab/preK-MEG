#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Define ROI based on 2 Hz SNR averaged across all subjects.
"""

import os
import yaml
import numpy as np
from mayavi import mlab
import mne
from analysis.aux_functions import load_paths, load_params, load_inverse_params


# flags
mlab.options.offscreen = True
mne.cuda.init_cuda()
mne.viz.set_3d_backend('mayavi')
roi_freq = 4

# load params
brain_plot_kwargs, *_ = load_params()
inverse_params = load_inverse_params()

# config paths
_, subjects_dir, results_dir = load_paths()
chosen_constraints = ('{orientation_constraint}-{estimate_type}'
                      ).format_map(inverse_params)

stc_dir = os.path.join(results_dir, 'pskt', 'group-level', 'stc',
                       chosen_constraints)
roi_dir = os.path.join('..', 'ROIs')
fig_dir = os.path.join(results_dir, 'pskt', 'group-level', 'fig', 'roi')
for _dir in (fig_dir,):
    os.makedirs(_dir, exist_ok=True)

# config other
timepoints = ('pre', 'post')
subdivide_epochs = 5
subdiv = f'-{subdivide_epochs}_sec' if subdivide_epochs else ''

# load fsaverage source space
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)

# average the SNR data across pre/post timepoints
avg_stc = None
for timepoint in timepoints:
    fname = f'GrandAvg-{timepoint}_camp-pskt{subdiv}-fft-snr'
    fpath = os.path.join(stc_dir, fname)
    stc = mne.read_source_estimate(fpath, subject='fsaverage')
    if avg_stc is None:
        avg_stc = stc.copy()
    else:
        avg_stc.data += stc.data
        avg_stc.data /= 2
del stc

bin_idx = avg_stc.time_as_index(roi_freq)
hemi_n_verts = avg_stc.vertices[0].size
hz = f'{roi_freq}_Hz'

# plot stc
clim = dict(kind='percent', lims=(95, 99, 99.9))
brain = avg_stc.plot(subject='fsaverage', clim=clim, initial_time=roi_freq,
                     **brain_plot_kwargs)
# define ROI labels
for threshold in np.linspace(1.5, 2.5, 11):
    labels = dict()
    for hemi in ('lh', 'rh'):
        data = (avg_stc.data[:hemi_n_verts] if hemi == 'lh' else
                avg_stc.data[hemi_n_verts:])
        verts = np.where(data[:, bin_idx] >= threshold)[0]
        label = mne.Label(verts, hemi=hemi, subject=avg_stc.subject)
        labels[hemi] = label.fill(fsaverage_src)
        fname = f'{chosen_constraints}-{hz}-SNR_{threshold:3.1f}-{hemi}'
        labels[hemi].save(os.path.join(roi_dir, fname))
    # save label verts also as YAML for easy dataframe filtering
    label_verts = {key: val.vertices.tolist() for key, val in labels.items()}
    fname = f'{hz}-SNR_{threshold:3.1f}.yaml'
    with open(os.path.join(roi_dir, fname), 'w') as outfile:
        yaml.dump(label_verts, outfile)
    # add labels to brain
    for hemi in ('lh', 'rh'):
        brain.add_label(labels[hemi], borders=False, color='c')
    fname = (f'GrandAvg-pre_and_post_camp-pskt{subdiv}-fft-snr-'
             f'{chosen_constraints}-{hz}-thresh_{threshold:3.1f}.png')
    fpath = os.path.join(fig_dir, fname)
    brain.save_image(fpath)
    brain.remove_labels()
