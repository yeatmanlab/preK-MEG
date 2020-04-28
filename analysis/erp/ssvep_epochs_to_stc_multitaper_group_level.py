#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Load SSVEP evokeds, compute multitaper spectra, apply inverse, morph,
aggregate across subjs, & compute PSD.
"""

import os
import numpy as np
import mne
from mne.time_frequency.multitaper import (_compute_mt_params, _mt_spectra,
                                           _psd_from_mt)
from aux_functions import (load_paths, load_params, load_psd_params,
                           load_cohorts)

# flags
mne.cuda.init_cuda()


# config paths
data_root, subjects_dir, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'evoked')
stc_dir = os.path.join(results_dir, 'pskt', 'group-level', 'stc')
os.makedirs(stc_dir, exist_ok=True)

# load params
_, _, subjects = load_params()
psd_params = load_psd_params()
intervention_group, letter_knowledge_group = load_cohorts()
groups = dict(GrandAvg=subjects)
groups.update(intervention_group)
groups.update(letter_knowledge_group)

# config other
timepoints = ('pre', 'post')
snr = 3.
lambda2 = 1. / snr ** 2
smoothing_steps = 10
bandwidth = psd_params['bandwidth']
subdivide_epochs = psd_params['epoch_dur']
subdiv = f'-{subdivide_epochs}_sec' if subdivide_epochs else ''

# for morph to fsaverage
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)
fsaverage_vertices = [s['vertno'] for s in fsaverage_src]


# loop over timepoints
for timepoint in timepoints:
    # loop over cohort groups
    for group, members in groups.items():
        # only do pretest knowledge comparison for pre-camp timepoint
        if group.endswith('Knowledge') and timepoint == 'post':
            continue
        # only do intervention cohort comparison for post-camp timepoint
        if group.endswith('Intervention') and timepoint == 'pre':
            continue

        all_spectra = list()
        all_eigvals = None
        # loop over group members
        for subj in members:
            fname = f'{subj}-{timepoint}_camp-pskt{subdiv}-ave.fif'
            evoked = mne.read_evokeds(os.path.join(in_dir, fname))
            assert len(evoked) == 1
            evoked = evoked[0]
            sfreq = evoked.info['sfreq']
            # do multitaper estimation
            mt_kwargs = dict(n_times=len(evoked.times), sfreq=sfreq,
                             bandwidth=bandwidth, low_bias=True,
                             adaptive=False)
            dpss, eigvals, adaptive = _compute_mt_params(**mt_kwargs)
            # eigvals should only depend on taper params, so it should be
            # the same for all subjects. Let's make sure:
            if all_eigvals is None:
                all_eigvals = eigvals
            else:
                assert all_eigvals == eigvals
            mt_spectra, freqs = _mt_spectra(evoked.data, dpss, sfreq)
            # convert to fake epochs object
            info = mne.create_info(evoked.ch_names, sfreq)
            mt_epochs = mne.EpochsArray(np.swapaxes(mt_spectra, 0, 1), info)
            # go to source space. inverse is only located in the ERP folder
            # tree, not in PSKT (TODO: this may change at some point)
            inv_path = os.path.join(data_root, f'{timepoint}_camp', 'twa_hp',
                                    'erp', subj, 'inverse',
                                    f'{subj}-80-sss-meg-inv.fif')
            inverse = mne.minimum_norm.read_inverse_operator(inv_path)
            stc = mne.minimum_norm.apply_inverse_epochs(
                mt_epochs, inverse, lambda2, pick_ori='normal',
                nave=evoked.nave)
            del evoked, mt_epochs
            # morph to fsaverage
            # stc is a list (from tapers) so create morph with stc[0]
            morph = mne.compute_source_morph(stc[0], subject_from=subj.upper(),
                                             subject_to='fsaverage',
                                             subjects_dir=subjects_dir,
                                             spacing=fsaverage_vertices,
                                             smooth=smoothing_steps)
            morphed_stcs = [morph.apply(s) for s in stc]
            del stc
            # extract the data from the STC
            all_spectra.append(morphed_stc.data)
        # all_spectra.shape will be
        # (n_subj, n_taper, n_vert, [n_xyz_components,] n_freq)...
        all_spectra = np.array(all_spectra)
        # ...but we need the second-to-last axis to be the tapers axis,
        # otherwise _psd_from_mt() won't work right
        all_spectra = np.moveaxis(all_spectra, 1, -2)
        # now it should be (n_subj, n_vert, [n_xyz_comp,] n_taper, n_freq)

        # baseline: average of magnitudes (no phase cancellation)
        weights = np.sqrt(all_eigvals)[np.newaxis, np.newaxis, :, np.newaxis]
        baseline_psd = _psd_from_mt(all_spectra, weights).mean(axis=0)

        # noise reduction (?): magnitude of the average
        nr_psd = _psd_from_mt(all_spectra.mean(axis=0), weights)

        # shape is now (n_vert, [n_xyz_components,] n_freq) which is suitable
        # for a [Vector]SourceEstimate, so we'll re-use the last one from our
        # loop:
        for kind, psd in dict(baseline=baseline_psd,
                              phase_cancelled=nr_psd).items():
            morphed_stc.data = psd
            morphed_stc.tstep = np.diff(freqs[:2])
            assert np.all(morphed_stc.times == freqs)
            fname = f'{group}-{timepoint}_camp-pskt{subdiv}-multitaper-{kind}'
            morphed_stc.save(os.path.join(stc_dir, fname))
