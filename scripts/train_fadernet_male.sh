#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:.

nohup python3 train/fadernet_model.py  \
    --gpu=3 \
    --train_batch=32 \
    --epochs=50 \
    --resize_choice=2 \
    --load_size=256 \
    --fine_size=256 \
    --train_dir=./datasets/celeba/Img/img_align_celeba \
    --celeba_attr_path=./datasets/celeba/Anno/list_attr_celeba.txt \
    --selected_attrs \
    Male \
    --attr \
    Male \
    --img_sz=256 \
    --img_fm=3 \
    --instance_norm=False \
    --init_fm=32 \
    --max_fm=512 \
    --n_layers=6 \
    --n_skip=0 \
    --deconv_method="convtranspose" \
    --hid_dim=512 \
    --dec_dropout=0 \
    --lat_dis_dropout=0.3 \
    --n_lat_dis=1 \
    --n_ptc_dis=0 \
    --n_clf_dis=0 \
    --smooth_label=0.2 \
    --lambda_ae=1 \
    --lambda_lat_dis=0.0001 \
    --lambda_ptc_dis=0 \
    --lambda_clf_dis=0 \
    --lambda_schedule=500000 \
    --clip_grad_norm=5 \
    --display_fre=400 \
    --save_fre=1 \
    --visdom_env=fadernet_male \
    --save_dir=./checkpoints/fadernet/male >log_fadernet_male &
