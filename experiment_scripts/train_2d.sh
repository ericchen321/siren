#!/bin/bash

img_name="$1"
exp_category="$2"
num_steps="$3"
point_batch_size="$4"
eval_patch_size="$5"
steps_til_ckpt="$6"
steps_til_summary="$7"

python experiment_scripts/train_img.py \
--experiment_name ${exp_category}_${img_name} \
--model_type=sine \
--image_path data/2d/$img_name.jpg \
--num_steps $num_steps \
--point_batch_size $point_batch_size \
--eval_patch_size $eval_patch_size \
--steps_til_ckpt $steps_til_ckpt \
--steps_til_summary $steps_til_summary
