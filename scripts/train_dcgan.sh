#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:.

nohup python train/dcgan_model.py  \
    --gpu=3 \
    --last_epoch=0 \
    --train_batch=16 \
    --resize_choice=1 \
    --load_size=64 \
    --fine_size=64 \
    --display_fre=10 \
    --g_input_dim=100 \
    --d_input_dim=1 \
    --save_img_rows=6 \
    --save_img_cols=12 \
    --loss_choice=dcgan \
    --gp_choice=none \
    --train_dir=./datasets/ \
    --visdom_env=dcgan \
    --save_dir=./checkpoints/dcgan >log_dcgan &