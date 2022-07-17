#!/bin/bash

TASK_ID=$1

if [ $TASK_ID == 0 ]
then
    SHAPE_NAME="armadillo"
elif [ $TASK_ID == 1 ]
then
    SHAPE_NAME="asian_dragon"
elif [ $TASK_ID == 2 ]
then
    SHAPE_NAME="at-ot"
elif [ $TASK_ID == 3 ]
then
    SHAPE_NAME="beard_man"
elif [ $TASK_ID == 4 ]
then
    SHAPE_NAME="camera"
elif [ $TASK_ID == 5 ]
then
    SHAPE_NAME="cathedral"
elif [ $TASK_ID == 6 ]
then
    SHAPE_NAME="david"
elif [ $TASK_ID == 7 ]
then
    SHAPE_NAME="dragon"
elif [ $TASK_ID == 8 ]
then
    SHAPE_NAME="dragon_warrior"
elif [ $TASK_ID == 9 ]
then
    SHAPE_NAME="engine"
elif [ $TASK_ID == 10 ]
then
    SHAPE_NAME="gear_shift"
elif [ $TASK_ID == 11 ]
then
    SHAPE_NAME="guangong"
elif [ $TASK_ID == 12 ]
then
    SHAPE_NAME="lion"
elif [ $TASK_ID == 13 ]
then
    SHAPE_NAME="lunar_lander"
elif [ $TASK_ID == 14 ]
then
    SHAPE_NAME="ninjago_city"
elif [ $TASK_ID == 15 ]
then
    SHAPE_NAME="oak_tree"
elif [ $TASK_ID == 16 ]
then
    SHAPE_NAME="thai_statue"
else
    echo "Error! Task ID not recognized"
    exit 1
fi

source experiment_scripts/train_3d.sh $SHAPE_NAME 5_hidden_layers 200000 10000 5000
