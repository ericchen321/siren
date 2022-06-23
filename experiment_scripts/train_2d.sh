#!/bin/bash

exp_name="$1"
num_steps="$2"
point_batch_size="$3"
eval_patch_size="$4"
steps_til_ckpt="$5"
steps_til_summary="$6"

python experiment_scripts/train_img.py \
--experiment_name $exp_name \
--model_type=sine \
--image_path data/2d/$exp_name.jpg \
--num_steps $num_steps \
--point_batch_size $point_batch_size \
--eval_patch_size $eval_patch_size \
--steps_til_ckpt $steps_til_ckpt \
--steps_til_summary $steps_til_summary