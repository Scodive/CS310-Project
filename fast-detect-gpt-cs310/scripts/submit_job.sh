#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=a100
#SBATCH -J myFirstJob
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

echo "===== 作业开始! ====="
python model_test.py

# 作业结束通知
echo "===== 作业完成! ====="
date