#!/bin/bash

python main.py --mode train_multi_path \
               --model CPCNet \
               --channels 64 \
               --dataset RAVEN \
               --dataset-path /home/ryan/datasets/comparison-experiments/A-RAVEN-10000-resized/ \
               --train-configs in_distribute_four_out_center_single \
                               distribute_nine \
                               distribute_four \
                               in_center_single_out_center_single \
                               left_center_single_right_center_single \
                               up_center_single_down_center_single \
                               center_single \
               --train-configs-proportion 1 1 1 1 1 1 1 \
               --image-size 80 \
               --epoch-num 2 \
               --batch-size 32 \
               --num-workers 2 \
               --learning-rate 0.01 \
               --loss bce \
               --output-path /home/ryan/gitee/CPCNet/results/2023_09_18_21_20_38_train_multi_path_CPCNet_ch_64_bs_32_dp_A-RAVEN-10000-resized_ls_bce_sd_2023
