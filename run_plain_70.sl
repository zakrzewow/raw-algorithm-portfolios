#!/bin/bash
#SBATCH --job-name=run-plain-70
#SBATCH --account=mandziuk-lab
#SBATCH --partition=short
#SBATCH --nodelist=sr-[1-3]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --mem-per-cpu=4G
#SBATCH --time=07:59:59
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=01161643@pw.edu.pl
#SBATCH --output=/home2/faculty/gzakrzewski/raw-algorithm-portfolios/log/%x-%j.log

srun /home2/faculty/gzakrzewski/miniconda3/envs/SMAC/bin/python /home2/faculty/gzakrzewski/raw-algorithm-portfolios/py_plain_70.py
