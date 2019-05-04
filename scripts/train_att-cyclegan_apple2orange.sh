#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:.

nohup python train/att_cyclegan_model.py \
    --gpu=1 \
    --resize_choice=3 \
    --load_size=286 \
    --fine_size=256 \
    --flip=1 \
    --workers=4 \
    --save_img_rows=2 \
    --save_img_cols=4 \
    --g_norm_type=instance \
    --g_lr=1e-4 \
    --g_weightdecay=5e-5 \
    --d_norm_type=instance \
    --d_lr=1e-4 \
    --d_weightdecay=5e-5 \
    --epochs=100 \
    --train_att_epochs=30 \
    --display_fre=200 \
    --save_fre=10 \
    --visdom_port=8891 \
    --loss_choice=lsgan \
    --g_resnet_blocks=9 \
    --lambdaA=5 \
    --lambdaB=5 \
    --lambdaI=0 \
    --g_input_dim=3 \
    --d_input_dim=3 \
    --mask_tau=0.1 \
    --save_img_rows=2 \
    --save_img_cols=6 \
    --train_dir=./datasets/apple2orange/trainA \
    --train_target_dir=./datasets/apple2orange/trainB \
    --save_dir=./checkpoints/att-cyclegan/apple2orange256_maskT0.1_lambdaAB5 \
    --visdom_env=att-cyclegan_apple2orange256_maskT0.1_lambdaAB5 \
    --load_when_switch \
    --train_batch=1  > log_att-cyclegan_apple2orange256_maskT0.1_lambdaAB5 &