#!/bin/bash

shape_name="$1"
exp_category="$2"
num_steps="$3"
batch_size="$4"
steps_til_ckpt="$5"
steps_til_summary="$6"

python experiment_scripts/train_sdf.py \
--experiment_name ${exp_category}_${shape_name} \
--model_type sine \
--point_cloud_path data/3d/$shape_name.xyz \
--num_steps $num_steps \
--batch_size $batch_size \
--steps_til_ckpt $steps_til_ckpt \
--steps_til_summary $steps_til_summary
