#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Load SSVEP evokeds, compute multitaper spectra, apply inverse, morph,
aggregate across subjs, & compute PSD.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.time_frequency.multitaper import (_compute_mt_params, _mt_spectra,
                                           _psd_from_mt)
from analysis.aux_functions import (load_paths, load_params, load_psd_params,
                           load_cohorts, subdivide_epochs)

# flags
mne.cuda.init_cuda()


# config paths
data_root, subjects_dir, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'epochs')
stc_dir = os.path.join(results_dir, 'pskt', 'group-level', 'stc')
fig_dir = os.path.join(results_dir, 'pskt', 'group-level', 'fig', 'phase')
for _dir in (stc_dir, fig_dir):
    os.makedirs(_dir, exist_ok=True)

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
trial_dur = 20
bandwidth = psd_params['bandwidth']
divisions = trial_dur // psd_params['epoch_dur']
subdiv = f"-{psd_params['epoch_dur']}_sec" if divisions > 1 else ''

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

        all_spectra = list()
        all_eigvals = None
        # loop over group members
        for subj in members:
            fname = f'{subj}-{timepoint}_camp-pskt-epo.fif'
            epochs = mne.read_epochs(os.path.join(in_dir, fname))
            if divisions > 1:
                epochs = subdivide_epochs(epochs, divisions)
            evoked = epochs.average()
            assert evoked.nave == 12 * divisions
            sfreq = evoked.info['sfreq']
            del epochs
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
            stcs = mne.minimum_norm.apply_inverse_epochs(
                mt_epochs, inverse, lambda2, pick_ori='normal',
                nave=evoked.nave)
            del evoked, mt_epochs
            these_spectra = list()
            # morph to fsaverage
            has_morph = False
            for stc in stcs:
                if not has_morph:
                    morph = mne.compute_source_morph(
                        stc, subject_from=subj.upper(), subject_to='fsaverage',
                        subjects_dir=subjects_dir, spacing=fsaverage_vertices,
                        smooth=smoothing_steps)
                morphed_stc = morph.apply(stc)
                del stc
                # extract the data from the STC. Shape of these_spectra will be
                # (n_vert, [n_xyz_components,] n_freq), accumulating across new
                # first dimension n_tapers
                these_spectra.append(morphed_stc.data)
            all_spectra.append(np.array(these_spectra))
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
        nr_psd = _psd_from_mt(all_spectra.mean(axis=0), weights.mean(axis=0))

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
            # spin off a phase plot
            if kind == 'phase_cancelled':
                vert, time = morphed_stc.get_peak(
                    'lh', tmin=5.9, tmax=6.1, mode='pos', vert_as_index=True,
                    time_as_index=True)
                # these_spectra will be n_subj × n_taper
                these_spectra = all_spectra[:, vert, ..., time]
                magnitudes = np.abs(these_spectra)
                phases = np.angle(these_spectra)
                fig, ax = plt.subplots(subplot_kw=dict(polar=True))
                ax.plot(phases, magnitudes,  '.', alpha=0.5)
                fname = f'{fname}-phases.png'
                fig.savefig(os.path.join(fig_dir, fname))
