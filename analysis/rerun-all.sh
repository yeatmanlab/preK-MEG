#!/bin/sh
set -e
export DISPLAY=:1.0

cd preprocessing
python run_mnefun.py
python find-optimal-reject-thresh.py
python check-epoch-drop-counts.py
python plot-test-retest-correlations.py

cd ../erp
python prek_make_stcs.py
python prek_make_group_averages.py
python prek_do_contrasts.py
python prek_extract_ROI_time_courses.py
python plot-test-retest-correlations.py

cd ../final-figs
python plot-ROI-label.py
python erp-spatial-and-temporal-ROIs.py
python erp-postcamp-lineplot-by-intervention-and-condition.py
python erp-lineplot-grid-intervention-by-condition-by-timepoint.py
python erp-rejection-thresholds.py
