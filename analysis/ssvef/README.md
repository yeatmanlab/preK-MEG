# Analysis pipeline for steady-state responses

1. `ssvep_make_epochs.py`
    - optional: `ssvep_plot_phases.py`
    - optional: `ssvep_plot_sensor_psds.py`

2. `ssvep_epochs_to_stc.py` applies the FFT to the epoch data and then converts
   to source space (keeping the complex values).
    - `ssvep_plot_stcs.py` plots each subject at 2, 4, 6, 12 Hz, both
      "magnitude" and "SNR" versions.

3. `ssvep_group_level_aggregate_stcs.py` aggregates individual subject STCs
   into group-level averages, in "magnitude", "SNR", and "log(SNR)" versions.
    - `ssvep_group_level_plot_stcs.py` plots group-average STCs at 2, 4, 6,
      and 12 Hz.

4. `ssvep_stats.py` computes uncorrected t-value maps and optionally runs
   clustering / TFCE.
    - `ssvep_plot_tvals.py` plots the t-value maps
    - `ssvep_plot_clusters.py` plots the clusters
