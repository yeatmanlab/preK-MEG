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
from analysis.aux_functions import load_paths, load_params


# flags
mlab.options.offscreen = True
mne.cuda.init_cuda()
mne.viz.set_3d_backend('mayavi')

# load params
brain_plot_kwargs, *_ = load_params()

# config paths
_, subjects_dir, results_dir = load_paths()

# TODO local testing
subjects_dir = '/data/prek/anat'
results_dir = '/data/prek/results'
brain_plot_kwargs.update(subjects_dir=subjects_dir)
# TODO end local testing

stc_dir = os.path.join(results_dir, 'pskt', 'group-level', 'stc')
roi_dir = os.path.join('..', 'ROIs')
fig_dir = os.path.join(results_dir, 'pskt', 'group-level', 'fig', 'brain')
for _dir in (fig_dir,):
    os.makedirs(_dir, exist_ok=True)

# config other
timepoints = ('pre', 'post')
trial_dur = 20
subdivide_epochs = 5
subdiv = f'-{subdivide_epochs}_sec' if subdivide_epochs else ''

# load fsaverage source space
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)

# container
avg_stc = None

# loop over timepoints
for timepoint in timepoints:
    # load SNR data
    fname = f'GrandAvg-{timepoint}_camp-pskt{subdiv}-fft-snr'
    fpath = os.path.join(stc_dir, fname)
    stc = mne.read_source_estimate(fpath, subject='fsaverage')
    if avg_stc is None:
        avg_stc = stc.copy()
    else:
        avg_stc.data += stc.data
        avg_stc.data /= 2
del stc


two_hz_idx = avg_stc.time_as_index(2.)
hemi_n_verts = avg_stc.vertices[0].size
# define ROI labels (a "generous" one and a smaller one)
for threshold in (2, 2.2):
    labels = dict()
    for hemi in ('lh', 'rh'):
        data = (avg_stc.data[:hemi_n_verts] if hemi == 'lh' else
                avg_stc.data[hemi_n_verts:])
        verts = np.where(data[:, two_hz_idx] >= threshold)[0]
        label = mne.Label(verts, hemi=hemi, subject=avg_stc.subject)
        labels[hemi] = label.fill(fsaverage_src)
        fname = f'2_Hz-SNR_{threshold:3.1f}-{hemi}'
        labels[hemi].save(os.path.join(roi_dir, fname))
    # save label verts also as YAML for easy dataframe filtering
    label_verts = {key: val.vertices.tolist() for key, val in labels.items()}
    fname = f'2_Hz-SNR_{threshold:3.1f}.yaml'
    with open(os.path.join(roi_dir, fname), 'w') as outfile:
        yaml.dump(label_verts, outfile)

    # plot stc
    clim = dict(kind='percent', lims=(95, 99, 99.9))
    brain = avg_stc.plot(subject='fsaverage', clim=clim, initial_time=2,
                         **brain_plot_kwargs)
    for hemi in ('lh', 'rh'):
        brain.add_label(labels[hemi], borders=False, color='c')
    fname = f'GrandAvg-pre_and_post_camp-pskt{subdiv}-fft-snr'
    fpath = os.path.join(fig_dir, f'{fname}-2_Hz-thresh_{threshold:3.1f}.png')
    brain.save_image(fpath)
    del brain
