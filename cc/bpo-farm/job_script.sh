#!/bin/bash

#SBATCH -A 
#SBATCH -t 00:03:00

# Default CPU-only job
#SBATCH --cpus-per-task 2
#SBATCH --mem 20G

# Nibi/Fir
# RGU: 3.48
# SBATCH --gpus-per-node H100-2g.20gb:1
# SBATCH --cpus-per-task 4
# SBATCH --mem 62G

# RGU: 1.74
# SBATCH --gpus-per-node H100-1g.10gb:1
# SBATCH --cpus-per-task 2
# SBATCH --mem 31G

# Narval
# RGU: 2
# SBATCH --gpus-per-node A100-3g.20gb:1
# SBATCH --cpus-per-task 6
# SBATCH --mem 62G
 
# RGU: 1.14
# SBATCH --gpus-per-node A100-2g.10gb:1
# SBATCH --cpus-per-task 3
# SBATCH --mem 31G

module load python/3.10
module load meta-farm

source ~/envs/bpo/bin/activate

task.run
