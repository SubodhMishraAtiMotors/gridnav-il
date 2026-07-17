#!/bin/bash
python scripts/generate_dataset.py \
  --num_demos 500 \
  --max_attempts 5000 \
  --out data/gridnav_pp_500_conn8_conservative_no_traversability_goalmask.npz \
  --seed 60 \
  --cost_to_go_normalization local \
  --include_goal_mask \
  --no_traversability_channel \
  --planner_connectivity 8

python scripts/train_cnn.py \
  --dataset data/gridnav_pp_500_conn8_conservative_no_traversability_goalmask.npz \
  --out_dir checkpoints/cnn_policy_500_conn8_conservative_no_traversability_goalmask_p15 \
  --epochs 60 \
  --batch_size 128 \
  --early_stopping_patience 15 \
  --min_delta 1e-5 \
  --save_every 5

python scripts/evaluate_checkpoint_sweep.py \
  --checkpoint_dir checkpoints/cnn_policy_500_conn8_conservative_no_traversability_goalmask_p15 \
  --out_dir outputs/sweep_cnn_policy_500_conn8_conservative_no_traversability_goalmask_p15 \
  --num_tests 100 \
  --seed 2000 \
  --cost_to_go_normalization local \
  --include_goal_mask \
  --no_traversability_channel \
  --planner_connectivity 8