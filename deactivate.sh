unset DAPS_CONDA_ENV
unset DAPS_DIR
unset PYTHONPATH

if hash module 2> /dev/null; then
  # Let's clean your environment
  if hash conda 2> /dev/null; then
    source deactivate
  fi
  module purge
fi
