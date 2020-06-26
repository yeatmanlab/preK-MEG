#!/bin/bash

conda activate rdev
echo "This will take a LONG TIME, will CHANGE THE ACTIVE CONDA ENVIRONMENT, and will OVERWRITE the input file!!!"
jupyter nbconvert --to notebook \
    --ExecutePreprocessor.kernel_name=ir \
    --ExecutePreprocessor.timeout=-1 \
    --log-level INFO \
    --inplace \
    --execute ssvep_ROI_modeling.ipynb
