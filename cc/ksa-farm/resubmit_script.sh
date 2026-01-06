#!/bin/bash
# This is an auxiliary job which runs after the farm is done. It is used
# for auto farm resubmission (the option -auto for submit.run), and for running an optional
# post-processing job (when the job script file final.sh is present in the root farm directory).

#  You have to replace Your_account_name below with the name of your account:
#SBATCH -A Your_account_name

#SBATCH -t 0-00:20
#SBATCH -N 1
#SBATCH -n 1

# Don't change anything below this line

module load meta-farm

autojob.run
