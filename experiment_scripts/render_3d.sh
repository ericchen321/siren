#!/bin/bash

shape_name="$1"
exp_category="$2"
logging_subdir="$3"
resolution="$4"

set -x
python experiment_scripts/test_sdf.py \
--experiment_name ${exp_category}_${shape_name} \
--checkpoint_path logs/$logging_subdir/${exp_category}_${shape_name}/checkpoints/model_final.pth \
--logging_root logs/$logging_subdir/ \
--resolution $resolution \
