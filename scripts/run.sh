#!/bin/bash
#
#SBATCH --job-name=job1
#SBATCH -e run_outputs/res_%j.err            # File to which STDERR will be written
#SBATCH --output=run_outputs/res_%j.txt     # output file
#
#SBATCH --ntasks=5
#SBATCH --mem=30GB
#
python main.py --config=../configs/config.yaml
# jupyter notebook --no-browser --port=8892
sleep 1
exit


