#!/bin/bash

shape_name="$1"
exp_category="$2"
logging_subdir="$3"
resolution="$4"

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

set -x
python experiment_scripts/test_sdf.py \
--experiment_name ${exp_category}_${shape_name} \
--num_hidden_layers ${num_hidden_layers} \
--checkpoint_path logs/$logging_subdir/${exp_category}_${shape_name}/checkpoints/model_final.pth \
--logging_root logs/$logging_subdir/ \
--resolution $resolution \
