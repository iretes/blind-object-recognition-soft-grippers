#!/bin/bash

cd "$(dirname "$0")/.."

mkdir ./results/SN

GPU_ID=${1:-0}

# Training on the closed set dataset
CUDA_VISIBLE_DEVICES=$GPU_ID python -m siamese_network.training \
--exp_descr "SN_closed_set" \
--dataset_path "./dataset/dataset.npz" \
--results_dir "./results/SN"

# Training on the base dataset
CUDA_VISIBLE_DEVICES=$GPU_ID python -m siamese_network.training \
--exp_descr "SN_base" \
--dataset_path "./dataset/base_dataset.npz" \
--results_dir "./results/SN"

# Evaluating 5-shot performance
CUDA_VISIBLE_DEVICES=$GPU_ID python -m  siamese_network.few_shot_eval \
--exp_descr "SN_5shot" \
--base_dataset_path "./dataset/base_dataset.npz" \
--novel_dataset_path "./dataset/5shot_dataset.npz" \
--results_dir "./results/SN" \
--model_checkpoint_path "./results/SN/SN_base/SN.pkl" \
--model_settings_path "./results/SN/SN_base/experiment_settings.json" \
--reference_dataset_size 60

# Evaluating 10-shot performance
CUDA_VISIBLE_DEVICES=$GPU_ID python -m  siamese_network.few_shot_eval \
--exp_descr "SN_10shot" \
--base_dataset_path "./dataset/base_dataset.npz" \
--novel_dataset_path "./dataset/10shot_dataset.npz" \
--results_dir "./results/SN" \
--model_checkpoint_path "./results/SN/SN_base/SN.pkl" \
--model_settings_path "./results/SN/SN_base/experiment_settings.json" \
--reference_dataset_size 120