# ----------------------------------------------------------------------------
# Set environment (making a mess in ur shell with respect XD)
#
# Usage:
# $ . activate.sh
#
# ----------------------------------------------------------------------------
daps_conda_env="deep-action-prop-gen"
daps_dirname=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

if hash module 2> /dev/null; then
  # Let's clean your environment
  if hash conda 2> /dev/null; then
    source deactivate
  fi
  module purge

  # Update environment variables with requirements (gcc, cuda, conda)
  module load compilers/cuda/7.0
  module load tools/cudnn/v3
  module load apps/conda

  # Activate conda enviroment
  if hash conda 2> /dev/null; then
    # Make sure the environment is loaded in "adversarial" conditions XD
    while true; do
      source activate $daps_conda_env
      if [ $? -eq 0 ]; then
        break
      else
        sleep $[ ( $RANDOM % 10 ) + 1 ]s
      fi
    done

    # We should have a module for that (in principle)
    export PYTHONPATH=$PYTHONPATH:$daps_dirname
  else
    echo "Conda was not loaded. Check your modules configuration."
  fi
else
  echo "Bro, I can't setup ur environemt. Do it yourself!"
fi
