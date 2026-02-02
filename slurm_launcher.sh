#!/bin/bash
#SBATCH --job-name=scalene_profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --output=scalene_%j.log
#SBATCH --error=scalene_%j.err

export PYTHONUNBUFFERED=1
scalene --cpu-only main.py