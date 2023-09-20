#!/bin/bash

python main.py --mode test_multi_path \
               --model CPCNet \
               --channels 64 \
               --dataset RAVEN \
               --dataset-path /home/ryan/datasets/comparison-experiments/A-RAVEN-10000-resized/ \
               --ckpt-path /home/ryan/gitee/CPCNet/results/2023_09_18_21_20_38_train_multi_path_CPCNet_ch_64_bs_32_dp_A-RAVEN-10000-resized_ls_bce_sd_2023/model_ckpt/ckpt-4   \
               --image-size 80 \
               --batch-size 32 \
               --num-workers 2 \
               --loss bce

