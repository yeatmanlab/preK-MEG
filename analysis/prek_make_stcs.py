#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:23:40 2019

@author: Maggie Clarke

Make orig and morphed source time course files for PreK project, create group
average and movie.

"""

import os
import mne
from os import path as op
from mne.minimum_norm import (apply_inverse, read_inverse_operator)

subj = ['prek_1951', 'prek_1921']

data_path = '/storage/Maggie/prek'
anat_path = mne.utils.get_subjects_dir(None, raise_error=True)
conditions = ['words', 'faces', 'cars', 'aliens']
method = "dSPM" # adjust to dSPM, sLORETA or eLORETA
snr = 3.
lambda2 = 1. / snr ** 2
smoothing_steps = 5 
kwargs = dict(views=['lat', 'med'], hemi='split', size=(800, 800),
              colormap='cool')

# for morph to fsaverage
src_fname = anat_path + '/fsaverage/bem/fsaverage-ico-5-src.fif'
src = mne.read_source_spaces(src_fname)
fsave_vertices = [s['vertno'] for s in src]

for si, s in enumerate(subj):
    source = mne.read_source_spaces(op.join(anat_path, '%s' % s, 'bem',
                                                '%s-oct-6-src.fif' % s))
    verts_from = [source[0]['vertno'], source[1]['vertno']]    
    fwd = mne.read_forward_solution(op.join(data_path, '%s' % s, 'forward',
                                            '%s-sss-fwd.fif' % s))
    cov = mne.read_cov(op.join(data_path, '%s' % s, 'covariance',
                               '%s-80-sss-cov.fif' % s))
    inv = read_inverse_operator(op.join(data_path, '%s' % s, 'inverse',
                                        '%s-80-sss-meg-inv.fif' % s))
    evokeds = [mne.read_evokeds(op.join(data_path, '%s' %s,
                                'inverse', 
                                'Conditions_80-sss_eq_%s-ave.fif' % s), 
    condition=c) for c in conditions]
    stcs = [apply_inverse(e, inv, lambda2, method=method, 
                          pick_ori=None) for e in evokeds]
    if not op.isdir(op.join(data_path, '%s' %s, 'stc')):
        os.mkdir(op.join(data_path, '%s' %s, 'stc'))
    for j, stc in enumerate(stcs):
        stc.save(op.join(data_path, '%s' %s, 'stc',
                         '%s_' % s + evokeds[j].comment))
        morph = mne.compute_source_morph(stc, s, 'fsaverage', 
                                         spacing=fsave_vertices, smooth=5)
        morphed_stcs = [morph.apply(ss) for ss in stcs]
        for j, stc in enumerate(morphed_stcs):
            stc.save(op.join(data_path, '%s' %s, 'stc',
                             '%s_%s_fsaverage_'
                             % (s, method) + evokeds[j].comment))
# make group average + movie
avg = 0

for c in conditions:
    for s in subj:
        avg += mne.read_source_estimate(op.join(data_path, '%s' %s, 'stc',
                                        '%s_%s_fsaverage_%s-lh.stc' 
                                         % (s, method, c)))                                       
    avg /= len(subj) 
    avg.save(op.join(data_path, 'fsaverage_%s_%s_GrandAvg_N%d.stc'
                     % (method, c, len(subj))))
    brain = avg.plot(subject ='fsaverage', **kwargs)
    outname = op.join(data_path,'fsaverage_%s_%s_GrandAvg_N%d.mov' 
                      % (method, c, len(subj)))
    brain.save_movie(outname, framerate=30, time_dilation=25,
                     interpolation='linear')  
