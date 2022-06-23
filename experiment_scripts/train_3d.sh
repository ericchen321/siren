#!/bin/bash

exp_name="$1"
num_steps="$2"
batch_size="$3"
steps_til_ckpt="$4"
steps_til_summary="$5"

python experiment_scripts/train_sdf.py \
--experiment_name $exp_name \
--model_type sine \
--point_cloud_path data/3d/$exp_name.xyz \
--num_steps $num_steps \
--batch_size $batch_size \
--steps_til_ckpt $steps_til_ckpt \
--steps_til_summary $steps_til_summary