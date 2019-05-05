# GAN 
## Image Generation
### DCGAN 
[Arxiv](https://arxiv.org/pdf/1511.06434.pdf)
### WGAN
[Arxiv](https://arxiv.org/pdf/1701.07875.pdf)
### WGAN-GP
[Arxiv](https://arxiv.org/pdf/1704.00028.pdf)
### BEGAN
[Arxiv](https://arxiv.org/pdf/1703.10717.pdf)
### SAGAN
[Arxiv](https://arxiv.org/pdf/1805.08318.pdf)

### Experiments on Cifar-10
![Image](https://github.com/innerlee/ganbase/raw/lxh2/res/cifar10.jpg)
### Experiments on ImageNet
![Image](https://github.com/innerlee/ganbase/raw/lxh2/res/imagenet.jpg)
### Experiments on CelebA
![Image](https://github.com/innerlee/ganbase/raw/lxh2/res/celeba.jpg)

## Style Transfer
### pix2pix
[Arxiv](https://arxiv.org/pdf/1611.07004.pdf)
#### pix2pix on Cityscapes
![Image](https://github.com/innerlee/ganbase/raw/lxh2/res/pix2pix.jpg)
### CycleGAN
[Arxiv](https://arxiv.org/pdf/1703.10593.pdf)
### Attention-CycleGAN
[Arxiv](https://arxiv.org/pdf/1806.02311.pdf)
#### CycleGAN vs Attention-CycleGAN on horse2zebra
![Image](https://github.com/innerlee/ganbase/raw/lxh2/res/horse2zebra.jpg)
#### CycleGAN vs Attention-CycleGAN on orange2apple
![Image](https://github.com/innerlee/ganbase/raw/lxh2/res/orange2apple.jpg)
### StarGAN
[Arxiv](https://arxiv.org/pdf/1711.09020.pdf)
#### StarGAN on CelebA
![Image](https://github.com/innerlee/ganbase/raw/lxh2/res/stargan.jpg)
### FaderNet
[Arxiv](https://arxiv.org/pdf/1706.00409.pdf)
#### FaderNet on CelebA
![Image](https://github.com/innerlee/ganbase/raw/lxh2/res/fadernet.jpg)
#### StarGAN vs FaderNet on CelebA
![Image](https://github.com/innerlee/ganbase/raw/lxh2/res/fadernet_vs_stargan.jpg)
## Train
### DCGAN
```
python train/dcgan_model.py --gpu=0 --last_epoch=0 --train_batch=16 --resize_choice=1 --load_size=64 --fine_size=64 --display_fre=10 dim=100 --d_input_dim=1 --save_img_rows=6 --save_img_cols=12  loss_choice=dcgan --gp_choice=none --train_dir=./datasets/ --visdom_env=dcgan --save_dir=./checkpoints/dcgan
```
### WGAN
```
python train/dcgan_model.py --gpu=0 --last_epoch=0 --train_batch=16 --resize_choice=1 --load_size=64 --fine_size=64 --display_fre=10 dim=100 --d_input_dim=1 --save_img_rows=6 --save_img_cols=12  loss_choice=wgan --gp_choice=none --train_dir=./datasets/ --visdom_env=wgan --save_dir=./checkpoints/wgan
```
### WGAN-GP
```
python train/dcgan_model.py --gpu=0 --last_epoch=0 --train_batch=16 --resize_choice=1 --load_size=64 --fine_size=64 --display_fre=10 dim=100 --d_input_dim=1 --save_img_rows=6 --save_img_cols=12  loss_choice=wgan --gp_choice=wgan-gp --train_dir=./datasets/ --visdom_env=wgan-gp --save_dir=./checkpoints/wgan-gp
```
### BEGAN
```
python train/began.py --gpu=0 --last_epoch=0 --train_batch=16 --resize_choice=1 --load_size=64 --fine_size=64 --display_fre=10 --g_input_dim=100 --d_input_dim=1 --save_img_rows=6 --save_img_cols=12 --loss_choice=began --gp_choice=none --train_dir=./datasets/ --visdom_env=began --save_dir=./checkpoints/began
```
### SAGAN
```
python train/sagan.py --gpu=0 --last_epoch=0 --train_batch=16 --resize_choice=1 --load_size=64 --fine_size=64 --display_fre=10 --g_input_dim=100 --d_input_dim=1 --save_img_rows=6 --save_img_cols=12 --loss_choice=wgan --gp_choice=none --train_dir=./datasets/ --visdom_env=sagan --save_dir=./checkpoints/sagan
```
### pix2pix
```
python train/pix2pix_model.py --gpu=0 --resize_choice=3 --flip=1 --workers=4 --save_img_rows=2 --save_img_cols=4 --g_norm_type=instance --g_lr=2e-4 --g_weightdecay=5e-5 --d_norm_type=instance --d_lr=2e-4 --d_weightdecay=5e-5 --display_fre=40 --save_fre=20 --lambdaI=0.5 --visdom_port=8891 --loss_choice=lsgan --g_resnet_blocks=9 --train_dir=./datasets/cityscapes/train --save_dir=./checkpoints/pix2pix/cityscapes256_lambdaI100_batch4 --train_batch=4 --visdom_env=pix2pix_cityscapes256_lambdaI100_batch4 --g_input_dim=3 --d_input_dim=6
```
### CycleGAN
```
python train/cyclegan_model.py --gpu=0 --resize_choice=3 --flip=1 --workers=4 --save_img_rows=2 --save_img_cols=4 --g_norm_type=instance --g_lr=2e-4 --g_weightdecay=5e-5 --d_norm_type=instance --d_lr=2e-4 --d_weightdecay=5e-5 --display_fre=40 --save_fre=20 --lambdaA=10 --lambdaB=10 --lambdaI=0.5 --visdom_port=8891 --loss_choice=lsgan --g_resnet_blocks=9 --train_dir=./datasets/horse2zebra/trainA --train_target_dir=./datasets/horse2zebra/trainB --save_dir=./checkpoints/cyclegan/horse2zebra256_lambdaI0 --train_batch=1 --visdom_env=cyclegan_horse2zebra256_lambdaI0 --g_input_dim=3 --d_input_dim=3
```
### Attention-CycleGAN
```
python train/att_cyclegan_model.py --gpu=0 --resize_choice=3 --flip=1 --workers=4 --save_img_rows=2 --save_img_cols=4 --g_norm_type=instance --g_lr=2e-4 --g_weightdecay=5e-5 --d_norm_type=instance --d_lr=2e-4 --d_weightdecay=5e-5 --display_fre=20 --save_fre=10 --lambdaA=10 --lambdaB=10 --lambdaI=0 --visdom_port=8891 --loss_choice=lsgan --g_resnet_blocks=9 --train_dir=./datasets/horse2zebra/testA --train_target_dir=./datasets/horse2zebra/testB --save_dir=./checkpoints/att-cyclegan/horse2zebra256_maskT0.1 --train_batch=1 --visdom_env=att-cyclegan_horse2zebra256_maskT0.1 --g_input_dim=3 --d_input_dim=3 --mask_tau=0.1 --train_att_epochs=1 --load_when_switch
```
### StarGAN
```
python train/stargan_model.py --gpu=0 --load_size=128 --fine_size=128 --resize_choice=2 --flip=1 --workers=4 --save_img_rows=2 --save_img_cols=4 --g_lr=1e-4 --d_lr=1e-4 --visdom_port=8891 --loss_choice=wgan --g_resnet_blocks=6 --d_resnet_blocks=6 --train_dir=./datasets/celeba/Img/img_align_celeba --celeba_attr_path=./datasets/celeba/Anno/list_attr_celeba.txt --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young --save_dir=./checlpoints/stargan --train_batch=16 --visdom_env=stargan --epochs=150 --save_fre=5 --display_fre=10 --g_lr_decay_gamma=0.999 --d_lr_decay_gamma=0.999 --g_lr_decay_step=1 --d_lr_decay_step=1
```
### FaderNet
```
python train/fadernet_model.py --gpu=0 --save_dir=./checkpoints/fadernet --train_dir=./datasets/celeba/Img/img_align_celeba --celeba_attr_path=./datasets/celeba/Anno/list_attr_celeba.txt --img_sz=256 --img_fm=3 --instance_norm=False --init_fm=32 --max_fm=512 --n_layers=6 --n_skip=0 --deconv_method="convtranspose" --hid_dim=512 --dec_dropout=0 --lat_dis_dropout=0.3 --n_lat_dis=1 --n_ptc_dis=0 --n_clf_dis=0 --smooth_label=0.2 --lambda_ae=1 --lambda_lat_dis=0.0001 --lambda_ptc_dis=0 --lambda_clf_dis=0 --lambda_schedule=500000 --clip_grad_norm=5 --display_fre=10 --save_fre=1 --train_batch=32 --selected_attrs Young --attr Young --visdom_env=fadernet_young
```
## Test
### DCGAN / WGAN / WGAN-GP / BEGAN / SAGAN
```
python test/test_image_generation.py 
```
### pix2pix / CycleGAN / Attention-CycleGAN
```
python test/test_style_transfer.py 
```
### StarGAN
```
python test/test_stargan.py
```
### FaderNet
```
python test/test_fadernet.py
```

