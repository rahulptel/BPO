#!/usr/bin/env
# coding: utf-8

# global vars
VENVS_DIR="/home/rahulpat/envs/"
VENV_NAME="bpo"

# load module
echo "Load module..."
module purge
module load cuda
module load python/3.12
module load gurobi/11.0.3

# create virtual env
if [ ! -d "./$VENVS_DIR/$VENV_NAME" ]; then
  echo "Create venv..."
  # create source
  virtualenv --no-download $VENVS_DIR/$VENV_NAME
  source $VENVS_DIR/$VENV_NAME/bin/activate
  echo ""

  # pip install
  echo "Install python packages..."
  pip install --no-index --upgrade pip
  pip install --no-index botorch
  pip install --no-index pandas
  pip install --no-index tensorboard
  pip install --no-index hydra-core
  pip install --no-index matplotlib
  pip install --no-index gurobipy
# activate virtual env
else
  echo "Activate venv..."
  source $VENVS_DIR/$VENV_NAME/bin/activate

fi
echo ""