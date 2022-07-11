#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --account=def-rhodin
#SBATCH --job-name=eval_cathedral_default_siren
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=128G
module load python/3.6
module load StdEnv/2020
module load cuda/11.0

cd /home/gxc321/
source SirenEnv/bin/activate
cd /home/gxc321/scratch/siren/
source experiment_scripts/render_3d.sh cathedral default 2022_07_10 1024
