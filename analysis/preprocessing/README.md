# Important things to check before (re)running preprocessing

- check the YAML config file (`mnefun_common_params.yml`) to see if
  `in_names`, `in_numbers`, `out_names`, and `out_numbers` are
  configured for "separate" or "joint" preprocessing ("joint" is
  where both the ERP and PSKT recordings are used to create a
  single set of blink/heartbeat projectors for all files).

- check `../../sswef_helpers/aux_functions.py` for the variable `PREPROCESS_JOINTLY`,
  which determines which input folder to look in for the raw data files
  (they are ultimately the same files, hard-linked between different directories,
  but this sets the mnefun "work_dir" so that the intermediate files end up in
  the right place).

- in `../../sswef_helpers/aux_functions.py` check the function `load_paths()`
  to see where the results dir will be (to make sure you're not overwriting
  past results, if that's not what you intend).

- check `../../params/current_cohort.yml` to see if you're set up to
  run the original group, replication group, or both together ('pooled').

# Script sequence

1. `run_mnefun.py`
2. `find-optimal-reject-thresh.py` → `crossval-results.csv`
3. `check-epoch-drop-counts.py` → `trial-counts-after-thresholding.csv`, `epoch-rejection-thresholds.yaml`, and `peak-to-peak-hists-and-rejection-thresholds.png`
4. `plot-test-retest-correlations.py` → `test-retest-label-timecourses.png`
