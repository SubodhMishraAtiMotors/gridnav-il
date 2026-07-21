#!/bin/bash
python scripts/generate_dataset.py \
  --num_demos 500 \
  --max_attempts 5000 \
  --out data/gridnav_pp_500_conn8_occupancy_goalmask_cost_encoding.npz \
  --seed 90 \
  --cost_to_go_normalization local \
  --include_goal_mask \
  --include_occupancy_channel \
  --include_cost_to_go_encoding \
  --cost_encoding_frequencies 1 2 4 8 \
  --no_traversability_channel \
  --no_clearance_channel \
  --planner_connectivity 8

python scripts/train_cnn.py \
  --dataset data/gridnav_pp_500_conn8_occupancy_goalmask_cost_encoding.npz \
  --out_dir checkpoints/cnn_policy_500_conn8_occupancy_goalmask_cost_encoding_p15 \
  --epochs 60 \
  --batch_size 128 \
  --early_stopping_patience 15 \
  --min_delta 1e-5 \
  --save_every 5

python scripts/evaluate_checkpoint_sweep.py \
  --checkpoint_dir checkpoints/cnn_policy_500_conn8_occupancy_goalmask_cost_encoding_p15 \
  --out_dir outputs/sweep_cnn_policy_500_conn8_occupancy_goalmask_cost_encoding_p15 \
  --num_tests 100 \
  --seed 2000 \
  --cost_to_go_normalization local \
  --include_goal_mask \
  --include_occupancy_channel \
  --include_cost_to_go_encoding \
  --cost_encoding_frequencies 1 2 4 8 \
  --no_traversability_channel \
  --no_clearance_channel \
  --planner_connectivity 8