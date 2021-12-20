#!/bin/sh
set -e
export DISPLAY=:1.0

python plot-ROI-label.py
python erp-spatial-and-temporal-ROIs.py
python erp-postcamp-lineplot-by-intervention-and-condition.py
python erp-lineplot-grid-intervention-by-condition-by-timepoint.py
python erp-rejection-thresholds.py
