#!/bin/bash
#Get execution dir
sh_dir=$(pwd)
echo "Executing from $sh_dir"
# Variables for keeping track of the experiment
experiment_root="data/debug-all/0/512-64"  # Folder to allocate outputs and experiment logs
experiment_id="000"
experiment_dir=$experiment_root/$experiment_id
# Variables regarding dataset
thumos_dir="data/thumos14"
c3d_hdf5_file="${thumos_dir}/c3d/val_c3d_pca.hdf5"

# Hyper-parameter of DAPs
K=64           # Number of anchors
T=512          # DAPs receptive field in terms of number of frames
rnn_time_steps=32  # Number of LSTM steps
f_stride_inference=64      # Sliding stride of DAPs during inference

# learning/model hyper-parameters
c3d_size=500
training_ratio=0.9
pooling_type=concat-${rnn_time_steps}-mean
depth=1
num_hidden_units=256
learning_rate=1e-4
alpha=1e-2
num_epochs=50
prefix_compute_priors_output=$experiment_root/thumos14
snapshot_freq=$(expr $num_epochs / 10)

# Finish if error
set -e

# Fun starts here!
# Create folder if it does not exist
if [ ! -d $experiment_root ]; then
   echo "Root folder for experiments was created ${experiment_root}"
   mkdir -p $experiment_root
fi

# Compute anchors/priors outputs
segments_csv=${prefix_compute_priors_output}"_ref.lst"
anchors_hdf5=${prefix_compute_priors_output}"_priors.hkl"
labels_hdf5=${prefix_compute_priors_output}"_conf.hkl"
# Skip if you already compute it
if [ ! -f $anchors_hdf5 ]; then
  echo "Compute priors"
  tools/compute_priors.py -k $K -t $T -rng 313 -d thumos14 -o $experiment_root/thumos14
fi

# Dump HDF5 used for training (it requires a bit of patience or optimization)
dataset_hdf5=$experiment_root"/val_fc7_${pooling_type}.hkl"
val_videos_csv=$experiment_root/val_videos.txt
# Skip if you already compute it
if [ ! -f $dataset_hdf5 ]; then
  echo "Create HDF5 for training"
  tools/create_dataset.py $segments_csv $c3d_hdf5_file $experiment_root -cf $labels_hdf5 -p $pooling_type -r $training_ratio -f2 -s -rng 2510 -v
  sed -i '1 i\video-name' $val_videos_csv
fi

# Learning
anchors_file_for_learning=$experiment_root/train_priors.hkl
last_model_npz=$experiment_dir/model.npz
hypprm_json=$experiment_dir/hyper_prm.json
# Skip if you already compute it
if [ ! -f $last_model_npz ]; then
  echo "Launch learning"
  # Stupid renaming (I should remove it)
  if [ ! -f $anchors_file_for_learning ]; then
    ln -s $sh_dir/$anchors_hdf5 $anchors_file_for_learning
  fi
  mkdir -p $experiment_dir
  tools/launch_learning.py --l_rate $learning_rate --alpha $alpha \
    --num_proposal $K --seq_length $rnn_time_steps --width $num_hidden_units \
    --depth $depth --model_type lstm --n_epoch $num_epochs \
    --opt_rule adagrad --batch_size 4096 --grad_clip 25 \
    --ds_suffix $pooling_type --output_dir $experiment_root
    --ds_prefix $experiment_root \
    --snapshot_freq $snapshot_freq -rng 12345 \
    --serial_jobs --debug -v
fi

# Evaluation
proposals_csv=$experiment_root/eval/0/result.proposals
if [ ! -f $proposals_csv ]; then
  echo "Launch detection"
  tools/daps_detection.py $last_model_npz $hypprm_json 0 $experiment_id \
    $experiment_root \
    --stride $f_stride_inference --T $T --no_nms 0.7 \
    --input_size $c3d_size --feat_file $c3d_hdf5_file \
    --pool_type $pooling_type --dataset thumos14-val \
    --file_filter $val_videos_csv -ow
fi
