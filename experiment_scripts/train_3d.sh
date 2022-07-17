#!/bin/bash

shape_name="$1"
exp_category="$2"
num_steps="$3"
steps_til_ckpt="$4"
steps_til_summary="$5"

if [ $exp_category == "default" ]
then
    num_hidden_layers=3
elif [ $exp_category == "4_hidden_layers" ]
then
    num_hidden_layers=4
elif [ $exp_category == "5_hidden_layers" ]
then
    num_hidden_layers=5
elif [ $exp_category == "6_hidden_layers" ]
then
    num_hidden_layers=6
elif [ $exp_category == "7_hidden_layers" ]
then
    num_hidden_layers=7
elif [ $exp_category == "8_hidden_layers" ]
then
    num_hidden_layers=8
else
    echo "Error! Exp category not recognized"
    exit 1
fi

batch_size=4096

python experiment_scripts/train_sdf.py \
--experiment_name ${exp_category}_${shape_name} \
--model_type sine \
--num_hidden_layers $num_hidden_layers \
--point_cloud_path data/3d/$shape_name.xyz \
--num_steps $num_steps \
--batch_size $batch_size \
--steps_til_ckpt $steps_til_ckpt \
--steps_til_summary $steps_til_summary
