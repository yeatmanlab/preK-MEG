# Important things to check before (re)running preprocessing

- check the YAML config file (`mnefun_common_params.yml`) to see if
  `in_names`, `in_numbers, `out_names`, and `out_numbers` are
  configured for "separate" or "joint" preprocessing ("joint" is
  where both the ERP and PSKT recordings are used to create a
  single set of blink/heartbeat projectors for all files).

- check `../analysis/aux_functions.py` for the variable `PREPROCESS_JOINTLY`,
  which determines which input folder to look in for the raw data files
  (they are ultimately the same files, hard-linked between different directories,
  but this sets the mnefun "work_dir" so that the intermediate files end up in
  the right place).

- check `../../params/current_cohort.yml` to see if you're set up to
  run the original group, replication group, or both together ('pooled').
