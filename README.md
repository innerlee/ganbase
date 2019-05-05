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
### pix2pix
```
python train/pix2pix_model.py --gpu=0 --resize_choice=3 --flip=1 --workers=4 --save_img_rows=2 --save_img_cols=4 --g_norm_type=instance --g_lr=2e-4 --g_weightdecay=5e-5 --d_norm_type=instance --d_lr=2e-4 --d_weightdecay=5e-5 --display_fre=40 --save_fre=20 --lambdaI=0.5 --visdom_port=8891 --loss_choice=lsgan --g_resnet_blocks=9 --train_dir=./datasets/cityscapes/train --save_dir=./checkpoints/pix2pix/cityscapes256_lambdaI100_batch4 --train_batch=4 --visdom_env=pix2pix_cityscapes256_lambdaI100_batch4 --g_input_dim=3 --d_input_dim=6
```
## Test
