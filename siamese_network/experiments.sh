#!/bin/bash

cd "$(dirname "$0")/.."

#mkdir ./results/SN

Training with default parameters
CUDA_VISIBLE_DEVICES=7 python -m siamese_network.training \
--exp_descr "SN_default" \
--dataset_path "./dataset/dataset.npz" \
--results_dir "./results/SN"


# CUDA_VISIBLE_DEVICES=7 python -m siamese_network.training \
# --exp_descr "SN_smaller_tau_longer_training" \
# --dataset_path "./dataset/dataset.npz" \
# --results_dir "./results/SN" \
# --tau 0.5 \
# --contrastive_learning_num_epochs 30
