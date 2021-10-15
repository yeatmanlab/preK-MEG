# Final figures

## ERP analysis

1. run `plot-ROI-label.py` first, to generate a brain with the ROI highlighted;
   this will get embedded in other figures later.

2. run `erp-spatial-and-temporal-rois.py` next; this generates the temporal
   ROI data in a YAML file, and also generates a figure showing the spatial and
   temporal ROIs and a grand mean waveform.

3. `erp-postcamp-lineplot-by-intervention-and-condition.py` generates a figure
   with the spatial ROI in one panel, and two lineplots (one per intervention)
   of three lines each (one per condition). The lineplots show post-camp data
   only.

4. `erp-lineplot-grid-intervention-by-condition-by-timepoint.py` generates a
   2 × 3 grid of lineplots (pre- vs post-camp for each condition in each
   intervention).

5. `erp-rejection-thresholds.py` generates a supplementary figure showing the
   results of the epoch rejection threshold grid search.

## SSVEP analysis
