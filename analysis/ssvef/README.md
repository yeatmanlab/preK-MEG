# Analysis pipeline for steady-state responses

1. `ssvep_make_epochs.py`
    - optional: `ssvep_plot_phases.py`
    - optional: `ssvep_plot_sensor_psds.py`

2. `ssvep_epochs_to_evoked_fft.py` averages the epoched data and applies FFT.


3. `ssvep_fft_evk_to_stc_fsaverage.py` converts the frequency-domain evokeds
   into source estimates (STCs), optionally looping over different inverse
   constraints (fixed/loose/free orientation; and vector/magnitude-only/
   normal-component-only retention).
    - optional: `ssvep_plot_stcs.py` plots each morphed subject at
      2, 4, 6, 12 Hz, both "amplitude" and "SNR" versions, at each combination
      of inverse constraints.

4. `ssvep_group_level_aggregate_stcs.py` aggregates individual subject STCs
   into group-level averages, in "amplitude" and "SNR" versions, at each
   combination of inverse constraints.
    - optional: `ssvep_group_level_plot_stcs.py` plots group-average STCs at
      2, 4, 6, and 12 Hz, at each combination of inverse constraints.

5. tval-based ROI creation:
    - `ssvep_prep_data_for_stats.py` aggregates signal & noise data
    - `ssvep_calc_tvals.py` computes uncorrected t-value maps
    - `ssvep_clustering.py` runs clustering on signal/noise data
    - `ssvep_plot_tvals.py` plots the t-value maps
    - `ssvep_plot_clusters.py` plots the clusters

6. 