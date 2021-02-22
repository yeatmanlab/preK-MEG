#!/bin/sh
export DISPLAY=:1.0

python prek_make_stcs.py && \
python prek_make_group_averages.py && \
python prek_do_contrasts.py && \
python prek_extract_ROI_time_courses.py && \
python prek_clustering.py && \
python prek_plot_clusters_quick_and_dirty.py #&& \
#python prek_plot_clusters.py && \
#bash prek_make_cluster_movie_from_stills.bash
