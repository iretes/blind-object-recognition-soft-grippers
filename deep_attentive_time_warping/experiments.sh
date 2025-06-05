#!/bin/bash

cd "$(dirname "$0")/.."

mkdir ./results/DATW/

GPU_ID=${1:-0}

# Training on the closed set dataset
CUDA_VISIBLE_DEVICES=$GPU_ID python -m deep_attentive_time_warping.training \
--exp_descr "DATW_closed_set" \
--dataset_path "./dataset/dataset.npz" \
--results_dir "./results/DATW"

# Training on the base dataset
CUDA_VISIBLE_DEVICES=$GPU_ID python -m deep_attentive_time_warping.training \
--exp_descr "DATW_base" \
--dataset_path "./dataset/base_dataset.npz" \
--results_dir "./results/DATW"

# Evaluating 5-shot performance
CUDA_VISIBLE_DEVICES=$GPU_ID python -m  deep_attentive_time_warping.few_shot_eval \
--exp_descr "DATW_5shot" \
--base_dataset_path "./dataset/base_dataset.npz" \
--novel_dataset_path "./dataset/5shot_dataset.npz" \
--results_dir "./results/DATW" \
--model_checkpoint_path "./results/DATW/DATW_base/DATW.pkl" \
--model_settings_path "./results/DATW/DATW_base/experiment_settings.json" \
--reference_dataset_size 60

# Evaluating 10-shot performance
CUDA_VISIBLE_DEVICES=$GPU_ID python -m  deep_attentive_time_warping.few_shot_eval \
--exp_descr "DATW_10shot" \
--base_dataset_path "./dataset/base_dataset.npz" \
--novel_dataset_path "./dataset/10shot_dataset.npz" \
--results_dir "./results/DATW" \
--model_checkpoint_path "./results/DATW/DATW_base/DATW.pkl" \
--model_settings_path "./results/DATW/DATW_base/experiment_settings.json" \
--reference_dataset_size 120