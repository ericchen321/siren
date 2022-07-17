#!/bin/bash

img_name="$1"
exp_category="$2"
num_steps="$3"
steps_til_ckpt="$4"
steps_til_summary="$5"

if [ $exp_category == "default" ]
then
    num_hidden_layers=3
    eval_patch_size=514288
elif [ $exp_category == "4_hidden_layers" ]
then
    num_hidden_layers=4
    eval_patch_size=131072
elif [ $exp_category == "5_hidden_layers" ]
then
    num_hidden_layers=5
    eval_patch_size=131072
elif [ $exp_category == "6_hidden_layers" ]
then
    num_hidden_layers=6
    eval_patch_size=131072
elif [ $exp_category == "7_hidden_layers" ]
then
    num_hidden_layers=7
    eval_patch_size=131072
elif [ $exp_category == "8_hidden_layers" ]
then
    num_hidden_layers=8
    eval_patch_size=131072
else
    echo "Error! Exp category not recognized"
    exit 1
fi

point_batch_size=1048576

python experiment_scripts/train_img.py \
--experiment_name ${exp_category}_${img_name} \
--model_type=sine \
--num_hidden_layers=$num_hidden_layers \
--image_path data/2d/$img_name.jpg \
--num_steps $num_steps \
--point_batch_size $point_batch_size \
--eval_patch_size $eval_patch_size \
--steps_til_ckpt $steps_til_ckpt \
--steps_til_summary $steps_til_summary
