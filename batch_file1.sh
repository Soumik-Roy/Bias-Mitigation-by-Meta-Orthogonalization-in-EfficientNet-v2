#!/bin/bash
#SBATCH --job-name=dlops-b20ai042
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8gb
#SBATCH --time=00:10:00
#SBATCH --output=terminalOutput_effnetResults_%j.log

sleep 2

module load python/3.8
sleep 2

python3.8 effnetResults.py
