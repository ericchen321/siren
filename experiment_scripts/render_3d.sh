#!/bin/bash

exp_name="$1"
batch_size="$2"
resolution="$3"

python experiment_scripts/test_sdf.py \
--experiment_name $exp_name \
--checkpoint_path logs/$exp_name/checkpoints/model_final.pth \
--batch_size $batch_size \
--resolution $resolution \
