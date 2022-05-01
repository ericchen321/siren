#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --account=def-rhodin
#SBATCH --job-name=trainTokyoSiren
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=24G
module load python/3.6
module load StdEnv/2020
module load cuda/11.0

cd /home/gxc321/
source SirenEnv/bin/activate
cd /home/gxc321/scratch/siren/
source experiment_scripts/train_2d_tokyo.sh