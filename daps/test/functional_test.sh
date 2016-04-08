#!/bin/bash
EXP_ROOT="data/experiments/tmp"
EXP_ID="001"
EXP_DIR=$EXP_ROOT/$EXP_ID
THUMOS_DIR="data/thumos14"

# Finish if error
set -e

SKIP_IF_FILE=$EXP_ROOT"/thumos14_priors.hkl"
if [ ! -s $SKIP_IF_FILE ]; then
  echo "Compute priors"
  python << END
from daps.data_generation import compute_priors, dump_files
from daps.datasets import Thumos14


T, n_priors = 256, 16
iou_thr, i_thr, rng_seed = 0.5, 1.0, 313
thumos_root = '$THUMOS_DIR'
prefix = '$EXP_ROOT'
th = Thumos14(thumos_root)
df_seg = th.segments_info()

# Generate segments for training and priors for regression
priors, df = compute_priors(df_seg, T, n_priors, iou_thr=iou_thr,
                            rng_seed=rng_seed)
dump_files(prefix + '/thumos14', df=df, priors=priors, conf=True)
END
fi

SKIP_IF_FILE=$EXP_ROOT"/val_fc7_concat-16-mean.hkl"
if [ ! -s $SKIP_IF_FILE ]; then
  echo "Create HDF5 for training"
  tools/create_dataset.py $EXP_ROOT/thumos14_ref.lst $THUMOS_DIR/c3d/val_c3d_pca.hdf5 $EXP_ROOT -cf $EXP_ROOT/thumos14_conf.hkl -p concat-16-mean -r 0.9 -f2 -s -rng 2510 -v
  sed -i '1 i\video-name' $EXP_ROOT/val_videos.txt
fi

SKIP_IF_FILE=$EXP_DIR"/model.npz"
if [ ! -s $SKIP_IF_FILE ]; then
  echo "Launch learning"
  if [ ! -s $EXP_ROOT/train_priors.hkl ]; then
    ln -s $(pwd)/$EXP_ROOT/thumos14_priors.hkl $EXP_ROOT/train_priors.hkl
  fi
  mkdir -p $EXP_DIR
  tools/launch_learning.py -io 1 -lr 1e-4 -a 0.1 -w 256 -d 1 -m lstm -ne 10 -sf 50 -ds concat-16-mean -s 120 -or adagrad -rng 261849 -sj -dg -bz 4096 -gc 25 -v -od $EXP_ROOT -dp $EXP_ROOT
fi

SKIP_IF_FILE=$EXP_DIR/eval/0/result.proposals
if [ ! -s $SKIP_IF_FILE ]; then
  echo "Launch detection"
  tools/daps_detection.py $EXP_DIR/model.npz $EXP_DIR/hyper_prm.json 0 $EXP_ID $EXP_ROOT -is 500 -dset thumos14-val -ff $EXP_ROOT/val_videos.txt -feat $THUMOS_DIR/c3d/val_c3d_pca.hdf5 -ow -pt concat-16-mean -s 128 -t 256 -nms -pr $EXP_ROOT/thumos14_priors.hkl
fi
