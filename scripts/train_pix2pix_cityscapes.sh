#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:.

nohup python train/pix2pix_model.py \
    --gpu=2 \
    --resize_choice=3 \
    --load_size=286 \
    --fine_size=256 \
    --flip=1 \
    --workers=4 \
    --save_img_rows=2 \
    --save_img_cols=4 \
    --g_norm_type=batch \
    --g_lr=2e-4 \
    --g_weightdecay=5e-5 \
    --d_norm_type=batch \
    --d_lr=2e-4 \
    --d_weightdecay=5e-5 \
    --epochs=200 \
    --display_fre=200 \
    --save_fre=20 \
    --visdom_port=8891 \
    --loss_choice=vanilla \
    --g_unet_depth=8 \
    --lambdaI=100 \
    --train_dir=./datasets/cityscapes/train \
    --save_dir=./checkpoints/pix2pix/cityscapes256_lambdaI100_batch4 \
    --train_batch=4 \
    --visdom_env=pix2pix_cityscapes256_lambdaI100_batch4 \
    --g_input_dim=3 \
    --d_input_dim=6 > log_pix2pix_cityscapes256_lambdaI100_batch4 &