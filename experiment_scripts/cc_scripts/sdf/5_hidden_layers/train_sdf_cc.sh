#!/bin/bash
#SBATCH --array=0-16
#SBATCH --time=08:00:00
#SBATCH --account=def-rhodin
#SBATCH --job-name=tr_sdf_5_hidden_layers_siren
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=24G
module load python/3.6
module load StdEnv/2020
module load cuda/11.0

cd /home/gxc321/
source SirenEnv/bin/activate
cd /home/gxc321/scratch/siren/

source experiment_scripts/cc_scripts/sdf/5_hidden_layers/train_sdf_per_task_cc.sh $SLURM_ARRAY_TASK_ID
