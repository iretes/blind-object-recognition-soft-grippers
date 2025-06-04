#!/bin/bash

cd "$(dirname "$0")/.."

#mkdir ./results/DATW/

# Training with default parameters
# CUDA_VISIBLE_DEVICES=7 python -m deep_attentive_time_warping.training \
# --exp_descr "DATW_default" \
# --dataset_path "./dataset/dataset.npz" \
# --results_dir "./results/DATW"

# CUDA_VISIBLE_DEVICES=2 python -m deep_attentive_time_warping.training \
# --exp_descr "DATW_smaller_tau" \
# --dataset_path "./dataset/dataset.npz" \
# --results_dir "./results/DATW" \
# --tau 0.5 \




# # Training with default parameters
# python -m deep_attentive_time_warping.training \
# --exp_descr "DATW_default" \
# --dataset_path "./dataset/dataset.npz" \
# --results_dir "./results"

# # TODO:
# # Training with tailored parameters
# python -m deep_attentive_time_warping.training \
# --exp_descr "DATW_smaller_tau" \
# --dataset_path "./dataset/dataset.npz" \
# --results_dir "./results" \
# --tau 0.5

# # Training with small U-Net architecture
# python -m deep_attentive_time_warping.training \
# --exp_descr "DATW_small_unet" \
# --dataset_path "./dataset/dataset.npz" \
# --results_dir "./results" \
# --small_unet

# # Training on easy base dataset TODO: scegli parametri e passa path
# python -m deep_attentive_time_warping.training \
# --exp_descr "DATW_base_easy" \
# --dataset_path "./dataset/base_dataset_easy.npz" \
# --results_dir "./results"

# # 5-shot evaluation, TODO: ref set
# python -m deep_attentive_time_warping.few_shot_eval \
# --exp_descr "DATW_5shot_easy" \
# --base_dataset_path "./dataset/base_dataset_easy.npz" \
# --novel_dataset_path "./dataset/5shot_dataset_easy.npz" \
# --results_dir "./results" \
# --model_checkpoint_path "./results/DATW_base_easy/DATW.pkl" \
# --model_settings_path "./results/DATW_base_easy/experiment_settings.json" \
# --reference_dataset_size 12*5

# # 10-shot evaluation, TODO: ref set
# python -m deep_attentive_time_warping.few_shot_eval \
# --exp_descr "DATW_10shot_easy" \
# --base_dataset_path "./dataset/base_dataset_easy.npz" \
# --novel_dataset_path "./dataset/10shot_dataset_easy.npz" \
# --results_dir "./results" \
# --model_checkpoint_path "./results/DATW_base_easy/DATW.pkl" \
# --model_settings_path "./results/DATW_base_easy/experiment_settings.json" \
# --reference_dataset_size 12*10

# # Training on random base dataset TODO: scegli parametri e passa path
# python -m deep_attentive_time_warping.training \
# --exp_descr "DATW_base_random" \
# --dataset_path "./dataset/base_dataset_random.npz" \
# --results_dir "./results"

# # 5-shot evaluation, TODO: ref set
# python -m deep_attentive_time_warping.few_shot_eval \
# --exp_descr "DATW_5shot_random" \
# --base_dataset_path "./dataset/base_dataset_random.npz" \
# --novel_dataset_path "./dataset/5shot_dataset_random.npz" \
# --results_dir "./results" \
# --model_checkpoint_path "./results/DATW_base_random/DATW.pkl" \
# --model_settings_path "./results/DATW_base_random/experiment_settings.json" \
# --reference_dataset_size 12*5

# # 10-shot evaluation, TODO: ref set
# python -m deep_attentive_time_warping.few_shot_eval \
# --exp_descr "DATW_10shot_random" \
# --base_dataset_path "./dataset/base_dataset_random.npz" \
# --novel_dataset_path "./dataset/10shot_dataset_random.npz" \
# --results_dir "./results" \
# --model_checkpoint_path "./results/DATW_base_random/DATW.pkl" \
# --model_settings_path "./results/DATW_base_random/experiment_settings.json" \
# --reference_dataset_size 12*10