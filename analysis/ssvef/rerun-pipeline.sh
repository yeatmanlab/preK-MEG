#!/bin/sh
export DISPLAY=:1.0
export MPL_BACKEND=Agg

python ssvep_make_epochs.py && \
python ssvep_epochs_to_evoked_fft.py && \
python ssvep_plot_sensor_psds.py &&              `# optional` \
python ssvep_plot_phases.py &&                   `# optional` \
python ssvep_fft_evk_to_stc_fsaverage.py && \
python ssvep_plot_stcs.py &&                     `# optional` \
python ssvep_group_level_aggregate_stcs.py && \
python ssvep_group_level_plot_stcs.py &&         `# optional` \
python ssvep_prep_data_for_stats.py && \
python ssvep_calc_tvals.py && \
python ssvep_plot_tvals.py && \
python ssvep_clustering.py && \
python ssvep_plot_clusters.py && \
python ssvep_to_dataframe.py && \
python ssvep_make_roi.py
