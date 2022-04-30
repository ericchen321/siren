#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=def-rhodin
#SBATCH --job-name=trainAlbertSiren
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=24G
module load python/3.6
module load StdEnv/2020
module load cuda/11.0

cd /home/gxc321/
source SirenEnv/bin/activate
cd /home/gxc321/scratch/siren/
source experiment_scripts/train_2d_albert.sh