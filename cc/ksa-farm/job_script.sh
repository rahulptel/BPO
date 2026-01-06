#!/bin/bash

#SBATCH -A $cc_account
#SBATCH -t 03:00:00
#SBATCH --cpus-per-task 2
#SBATCH --mem 16G

module load python/3.10
module load gurobi/11.0.3
module load scipoptsuite
module load meta-farm

source ~/envs/bpo/bin/activate
export BASEPATH="/home/rahulpat/scratch/BPO"

task.run
