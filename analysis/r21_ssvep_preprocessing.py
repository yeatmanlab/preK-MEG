# -*- coding: utf-8 -*-

# Authors: Sung Jun Joo; Jason Yeatman; Kambiz Tavabi <ktavabi@gmail.com>
#
#
# License: BSD (3-clause)

import numpy as np
import mnefun
import os
#import glob
os.chdir('/home/sjjoo/git/BrainTools/projects/NLR_MEG')
from score import score
from prek_organize import prek_organize
import mne
import time
#import pycuda.driver 
#import pycuda.autoinit

t0 = time.time()

#mne.set_config('MNE_USE_CUDA', 'true')

# At Possum projects folder mounted in the local disk
raw_dir = '/home/sjjoo/git/SSWEF/pilot/prek'

# At local hard drive
out_dir = '/mnt/scratch/r21'
out_dir = '/home/sjjoo/Documents/r21'

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

out = prek_organize(raw_dir, out_dir)
    
#%%
os.chdir(out_dir)
out = os.listdir()

for nn, ss in enumerate(out):  
    params = mnefun.Params(tmin=-0.1, tmax=40, n_jobs=18, t_adjust=-4e-3,
                       decim=2, n_jobs_mkl=1, proj_sfreq=250,
                       n_jobs_fir='cuda', n_jobs_resample='cuda',
                       filter_length='5s', epochs_type='fif', lp_cut=40.,
#                       hp_cut=0.15,hp_trans=0.1,
                       bmin=-0.1, auto_bad=20., plot_raw=False) 
#                       bem_type = '5120-5120-5120')
    params.subjects = [ss]
    params.sss_type = 'python'
    params.sss_regularize = 'in' # 'in' by default
    params.tsss_dur = 16. # 60 for adults with not much head movements. This was set to 6.
    params.st_correlation = 0.98
    
    params.auto_bad_meg_thresh = 10 # THIS SHOULD NOT BE SO HIGH!

    params.trans_to = 'median'
    
    #print("Running " + str(len(params.subjects)) + ' Subjects') 
#    print("\n\n".join(params.subjects))
    print("\n\n")
    print("Running " + str(params.subjects)) 
    print("\n\n")
    
    params.subject_indices = np.arange(0,len(params.subjects))
    
    params.structurals =[None] * len(params.subjects)
    
    if params.subjects[0] == 'prek_1451_190419':
        params.run_names = ['%s_01', '%s_02', '%s_03', '%s_04']
    else:
        params.run_names = ['%s_01', '%s_02', '%s_03', '%s_04', '%s_05']
    
    
    params.dates = [(2014, 0, 00)] * len(params.subjects)
    #params.subject_indices = [0]
    params.score = score  # scoring function to use
    params.plot_drop_logs = False
    params.on_missing = 'warning'
    #params.acq_ssh = 'kambiz@minea.ilabs.uw.edu'  # minea - 172.28.161.8
    #params.acq_dir = '/sinuhe/data03/jason_words'

#    params.acq_ssh = 'jason@minea.ilabs.uw.edu'  # minea - 172.28.161.8
#    params.acq_dir = '/sinuhe/data03/jason_words'
    params.sws_ssh = 'jason@kasga.ilabs.uw.edu'  # kasga - 172.28.161.8
    params.sws_dir = '/data05/jason/NLR'
    
    #params.mf_args = '-hpie 30 -hpig .8 -hpicons' # sjjoo-20160826: We are doing SSS using python
    
    # epoch  

    params.reject = dict(grad=3000e-13, mag=4.0e-12) 
#    params.reject = dict(grad=4000e-13, mag=4.0e-12)    

    params.ssp_eog_reject = dict(grad=params.reject['grad'], mag=params.reject['mag'], eog=np.inf)
    params.ssp_ecg_reject = dict(grad=params.reject['grad'], mag=params.reject['mag'], ecg=np.inf)
        
    params.flat = dict(grad=1e-13, mag=1e-15)
    
    params.auto_bad_reject = dict(grad=2*params.reject['grad'], mag=2*params.reject['mag'])         
        
    params.auto_bad_flat = params.flat
      
    params.cov_method = 'shrunk'
    
    params.get_projs_from = range(len(params.run_names))
    params.inv_names = ['%s']
    params.inv_runs = [range(0, len(params.run_names))]
    params.runs_empty = []
    
    params.proj_nums = [[0, 0, 0],  # ECG: grad/mag/eeg
                        [1, 1, 0],  # EOG # sjjoo-20160826: was 3
                        [0, 0, 0]]  # Continuous (from ERM)
    
    # The scoring function needs to produce an event file with these values
    params.in_names = ['start']
    
    params.in_numbers = [1]
    
    # These lines define how to translate the above event types into evoked files
    params.analyses = [
        'All'
        ]
    
    params.out_names = [
        ['ALL']
    ]
    
    params.out_numbers = [
        [1]
        ]
    
    params.must_match = [
        [],
        ]
    # Set what will run
    mnefun.do_processing(
        params,
        fetch_raw=False,     # Fetch raw recording files from acquisition machine
        do_score=False,      # Do scoring to slice data into trials
    
        # Before running SSS, make SUBJ/raw_fif/SUBJ_prebad.txt file with
        # space-separated list of bad MEG channel numbers
        push_raw=False,      # Push raw files and SSS script to SSS workstation
        do_sss=True,        # Run SSS remotely (on sws) or locally with mne-python
        fetch_sss=False,     # Fetch SSSed files from SSS workstation
        do_ch_fix=False,     # Fix channel ordering
    
        # Before running SSP, examine SSS'ed files and make
        # SUBJ/bads/bad_ch_SUBJ_post-sss.txt; usually, this should only contain EEG
        # channels.
        gen_ssp=False,       # Generate SSP vectors
        apply_ssp=False,     # Apply SSP vectors and filtering
#        plot_psd=False,      # Plot raw data power spectra
        write_epochs=False,  # Write epochs to disk
        gen_covs=False,      # Generate covariances
    
        # Make SUBJ/trans/SUBJ-trans.fif using mne_analyze; needed for fwd calc.
        gen_fwd=False,       # Generate forward solutions (and src space if needed)
        gen_inv=False,       # Generate inverses
        gen_report=False,    # Write mne report html of results to disk
        print_status=False,  # Print completeness status update
    )

print('%i sec' % (time.time() - t0,))
