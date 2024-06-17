#!/bin/bash

# #SBATCH --account=cs678sp23
#SBATCH --job-name=i2ttox
##SBATCH --partition=normal
##SBATCH --nodes 1
##SBATCH --ntasks 1
##SBATCH --time 2-00:00:00
##SBATCH --cpus-per-task 1
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=40GB
#SBATCH --time=0-06:00:00
#SBATCH --array=1-14
#SBATCH --output=/scratch/craj/logs/iat/out/i2ttox_n.%A_%a.out.txt
#SBATCH --error=/scratch/craj/logs/iat/err/i2ttox_n.%A_%a.err.txt

#nvidia-smi

# Read arguments from the config file
readarray -t ARGS < splits2.txt
ARGUMENTS=${ARGS[$SLURM_ARRAY_TASK_ID - 1]}

echo $ARGUMENTS

# Execute the script with the arguments
# python t2t_task_llama3_8b_i.py $ARGUMENTS
# accelerate launch \
#    --multi_gpu \
#    --num_processes 4 \
#    --mixed_precision fp16 \
#    t2t_task_llama3_8b_i.py $ARGUMENTS
accelerate launch i2t_toxicity_analysis.py $ARGUMENTS