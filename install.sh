#!/bin/bash
# ```
#
# Sample bash-script used to setup our package
#
# This script just create the conda environment and all our
# python-dependencies. It will not install all the dependencies
# such as gcc, cuda.
#
# Usage: ./install.sh OR sh install.sh
# Requirements: conda
#
# ```
set -e
# Shortcut in case you wanna use another name for the environment
daps_dirname=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
daps_conda_env=daps-eccv16

# check conda existence
if hash conda 2>/dev/null; then
  # check if environment exists
  if [ ! "$(conda env list | grep "$conda_env_name")" ]; then
    # Use appropriate YAML-file, so far just x64
    conda env create -f $daps_dirname/"environment-proto-linux-x64.yml"
    source activate $conda_env_name
  else
    source activate $conda_env_name
    conda env update -f $daps_dirname/"environment-proto-linux-x64.yml"
  fi

  # Install the same version of Lasagne/Theano used in our work
  pip install --upgrade --no-deps git+https://github.com/Theano/Theano.git@88648d8d5531deb4e5e4201a3663ffbc1465b84d
  pip install --upgrade --no-deps git+https://github.com/Lasagne/Lasagne.git@9886da26df40cbde9222d4e20706b4b21bbdb627
else
  echo "Conda is not installed"
  return -1
fi
