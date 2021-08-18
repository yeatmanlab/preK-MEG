#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:01:29 2019

@author: Maggie Clarke

Setup source surfaces, bem and source space for PreK project
"""
import os
import os.path as op
from shutil import copyfile
import mne
from analysis.aux_functions import load_paths, load_params

# load params
_, subjects_dir, _ = load_paths()
*_, subjects, cohort = load_params(experiment='erp')
surf_names = ['inner_skull', 'outer_skull', 'outer_skin']

print(f'running {cohort} cohort')
for subj in subjects:
    subj = subj.upper()
    print('  %s: making surfaces' % subj, end='')
    p = os.popen('mne_watershed_bem --subject %s --overwrite' % subj)
    print(p.read())
    print('  %s: making high res head surface' % subj, end='')
    p = os.popen('mne_make_scalp_surfaces --subject %s' % subj)
    print(p.read())
    for s in surf_names:
        orig = op.join(subjects_dir, subj, 'bem', 'watershed',
                       '%s_%s_surface' % (subj, s))
        new = op.join(subjects_dir, subj, 'bem', '%s.surf' % s)
        copyfile(orig, new)
    bem = mne.make_bem_model(subject=subj, conductivity=[0.3])
    sol = mne.make_bem_solution(bem)
    print('  %s: writing bem solution' % subj, end='')
    mne.write_bem_solution(op.join(subjects_dir, subj, 'bem',
                                   '%s-5120-bem-sol.fif' % subj), sol)
    src = mne.setup_source_space(subject=subj, spacing='oct6', n_jobs=18)
    print('  %s: writing source space' % subj, end='')
    mne.write_source_spaces(op.join(subjects_dir, subj, 'bem',
                                    '%s-oct-6-src.fif' % subj), src)
    bplt = mne.viz.plot_bem(subject=subj)
    bplt.savefig(op.join(subjects_dir, subj, 'bem',
                         '%s-bem_surf.png' % subj))
