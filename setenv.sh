# Script to setup enviroment variables
# Usage: . setenv.sh OR source setenv.sh
module purge
module load compilers/cuda/7.0
module load tools/cudnn/v3
# Activate conda enviroment
source activate deep-action-prop-gen

# Get project dir
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Add source to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$DIR/daps
