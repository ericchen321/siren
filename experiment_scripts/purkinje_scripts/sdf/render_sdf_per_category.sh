#!/bin/bash

exp_category="$1"
logging_subdir="$2"
resolution="$3"

declare -a shape_names=(
    "armadillo"
    "asian_dragon"
    "at-ot"
    "beard_man"
    "camera"
    "cathedral"
    "david"
    "dragon"
    "dragon_warrior"
    "engine"
    "gear_shift"
    "guangong"
    "lion"
    "lunar_lander"
    "ninjago_city"
    "oak_tree"
    "thai_statue"
    )

for shape_name in ${shape_names[@]}; do
    source experiment_scripts/render_3d.sh \
    $shape_name $exp_category $logging_subdir $resolution

    mv logs/$logging_subdir/${exp_category}_${shape_name}/test.ply \
    logs/$logging_subdir/${exp_category}_${shape_name}/${shape_name}_${resolution}.ply
done