#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Extract SSVEP epochs, downsample, and save to disk. Optionally generate
spectrum estimates and plot PSDs / scalp topographies for each subject.
"""

import os
import numpy as np
import mne
from mne.time_frequency.multitaper import (_compute_mt_params, _mt_spectra,
                                           _psd_from_mt)
from aux_functions import load_paths, load_params

# flags
mne.cuda.init_cuda()

# config paths
data_root, subjects_dir, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'epochs')
evk_dir = os.path.join(results_dir, 'pskt', 'evoked')
stc_dir = os.path.join(results_dir, 'pskt', 'stc', 'subject-specific')
morph_dir = os.path.join(results_dir, 'pskt', 'stc', 'morphed-to-fsaverage')
for _dir in (evk_dir, stc_dir, morph_dir):
    os.makedirs(_dir, exist_ok=True)

# # TODO local testing stuff
# in_dir = '/home/drmccloy/Desktop/prek/ssvep_tmp/epochs'
# evk_dir = '/home/drmccloy/Desktop/prek/ssvep_tmp/evoked'
# stc_dir = '/home/drmccloy/Desktop/prek/ssvep_tmp/'
# s = 'prek_1964'
# timepoint = 'pre'
# subjects_dir = mne.get_config('SUBJECTS_DIR')
# inv_path = '/home/drmccloy/Desktop/prek/ssvep_tmp/inverse/prek_1964-80-sss-meg-inv.fif'

# load params
_, _, subjects = load_params()

# config other
timepoints = ('pre', 'post')
snr = 3.
lambda2 = 1. / snr ** 2
smoothing_steps = 10

# for morph to fsaverage
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)
fsaverage_vertices = [s['vertno'] for s in fsaverage_src]


# loop over subjects
for s in subjects:
    already_morphed = False
    # loop over timepoints
    for timepoint in timepoints:
        # load epochs
        fname = f'{s}-{timepoint}_camp-pskt-epo.fif'
        fpath = os.path.join(in_dir, fname)
        epochs = mne.read_epochs(fpath, proj=True)
        # TODO separately for ps vs kt
        evoked = epochs.average()
        fname = f'{s}-{timepoint}_camp-pskt-ave.fif'
        evoked.save(os.path.join(evk_dir, fname))
        del epochs
        # do multitaper estimation
        sfreq = evoked.info['sfreq']
        mt_kwargs = dict(n_times=len(evoked.times), sfreq=sfreq,
                         bandwidth=0.2, low_bias=True, adaptive=False)
        dpss, eigvals, adaptive = _compute_mt_params(**mt_kwargs)
        mt_spectra, freqs = _mt_spectra(evoked.data, dpss, sfreq)
        # convert to fake epochs object
        info = mne.create_info(evoked.ch_names, sfreq)
        mt_epochs = mne.EpochsArray(np.swapaxes(mt_spectra, 0, 1), info)
        # go to source space. inverse is only located in the ERP folder tree,
        # not in PSKT (TODO: this may change at some point)
        inv_path = os.path.join(data_root, f'{timepoint}_camp', 'twa_hp',
                                'erp', s, 'inverse', f'{s}-80-sss-meg-inv.fif')
        inverse = mne.minimum_norm.read_inverse_operator(inv_path)
        stc = mne.minimum_norm.apply_inverse_epochs(
            mt_epochs, inverse, lambda2, pick_ori='vector', nave=evoked.nave)
        del evoked, mt_epochs
        # extract the data from STCs
        data = np.array([s.data for s in stc])
        # data.shape will be (n_tapers, n_vertices, n_xyz_components, n_freqs)
        # but we need the second-to-last axis to be the tapers axis, otherwise
        # _psd_from_mt() won't work right
        data = np.moveaxis(data, 0, -2)
        weights = np.sqrt(eigvals)[np.newaxis, np.newaxis, :, np.newaxis]
        # use one STC to hold the aggregated data, delete rest to save memory
        stc = stc[0]
        # combine the multitaper spectral estimates
        psd = _psd_from_mt(data, weights)
        stc.data = psd
        stc.tstep = np.diff(freqs[:2])
        assert np.all(stc.times == freqs)
        del psd
        fname = f'{s}-{timepoint}_camp-pskt-multitaper'
        stc.save(os.path.join(stc_dir, fname))
        # compute morph for this subject
        if not already_morphed:
            morph = mne.compute_source_morph(stc, subject_from=s.upper(),
                                             subject_to='fsaverage',
                                             subjects_dir=subjects_dir,
                                             spacing=fsaverage_vertices,
                                             smooth=smoothing_steps)
            already_morphed = True
        # morph and save
        morphed_stc = morph.apply(stc)
        fname = f'{s}FSAverage-{timepoint}_camp-pskt-multitaper'
        fpath = os.path.join(morph_dir, f'{timepoint}_camp', s)
        os.makedirs(fpath, exist_ok=True)
        morphed_stc.save(os.path.join(fpath, fname))
