#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:.

nohup python train/cyclegan_model.py \
    --gpu=3 \
    --resize_choice=3 \
    --load_size=286 \
    --fine_size=256 \
    --flip=1 \
    --workers=4 \
    --save_img_rows=2 \
    --save_img_cols=4 \
    --g_norm_type=instance \
    --g_lr=2e-4 \
    --g_weightdecay=5e-5 \
    --d_norm_type=instance \
    --d_lr=2e-4 \
    --d_weightdecay=5e-5 \
    --epochs=200 \
    --display_fre=200 \
    --save_fre=20 \
    --visdom_port=8891 \
    --loss_choice=lsgan \
    --g_resnet_blocks=9 \
    --lambdaA=10 \
    --lambdaB=10 \
    --lambdaI=0 \
    --g_input_dim=3 \
    --d_input_dim=3 \
    --train_dir=./datasets/apple2orange/trainA \
    --train_target_dir=./datasets/apple2orange/trainB \
    --save_dir=./checkpoints/cyclegan/apple2orange256_lambdaI0 \
    --visdom_env=cyclegan_apple2orange256_lambdaI0 \
    --train_batch=1  > log_cyclegan_apple2orange256_lambdaI0 &