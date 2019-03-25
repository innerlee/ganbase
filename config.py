# -*- coding: utf-8 -*-
import argparse


def train_config():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu', default='0', type=str, help='index of gpu to use, default 0')
    parser.add_argument('--train_batch', default=128, type=int, help='train batch size, default 128')
    parser.add_argument('--valid_batch', default=8, type=int, help='valid batch size, default 8')
    parser.add_argument('--resize_choice', default=2, type=int, help='0 for ori, 1 for resize, 2 for rand, 3 for cent')
    parser.add_argument('--load_size', default=286, type=int, help='load size, default 286')
    parser.add_argument('--fine_size', default=256, type=int, help='fine size, default 256')
    parser.add_argument('--workers', default=4, type=int, help='num of workers to use, default 4')
    parser.add_argument('--epochs', default=200, type=int, help='epochs to train and val, default 200')
    parser.add_argument('--flip', default=1, type=int, help='0 for not flip, 1 for hor flip, 2 for ver flip, default 1')
    parser.add_argument('--train_dir', type=str, help='dir for train data')
    parser.add_argument('--valid_dir', default=None, type=str, help='dir for valid data')
    parser.add_argument('--save_dir', type=str, help='dir to save model and results')
    parser.add_argument('--last_epoch', default=0, type=int, help='train with last epoch ,0 for restart')
    parser.add_argument('--save_img_rows', default=2, type=int, help='rows of image grid to save every epoch')
    parser.add_argument('--save_img_cols', default=4, type=int, help='cols of image grid to save every epoch')

    parser.add_argument('--init_type', default='normal', type=str, help='init type')
    parser.add_argument('--init_std', default=0.02, type=float)
    # TODO add more init choice
    # parser.add_argument('--init_mean', default=0., type=float)
    # parser.add_argument('--init_bound', default=-1, type=float)
    # parser.add_argument('--init_gain', default=1, type=float)
    # parser.add_argument('--init_a', default=1, type=float)
    # parser.add_argument('--init_mode', default='fin_in', type=str)
    # parser.add_argument('--init_nonlinearty', default='relu', type=str)

    parser.add_argument('--g_norm_type', default='instance', type=str, help='norm type for G')
    parser.add_argument('--g_lr', default=2e-4, type=float, help='learning rate for optimizer, default 2e-4')
    parser.add_argument('--g_lr_decay_step', default=30, type=int, help='decay step for lr')
    parser.add_argument('--g_lr_decay_gamma', default=0.1, type=float, help='decay ratio for lr')
    # TODO add optimizer choice
    parser.add_argument('--g_lr_policy', default='linear', type=str, help='[linear | lambda | step | plateau]')
    parser.add_argument('--g_beta1', default=0.5, type=float, help='beta1 for adam optimizer, default 0.5')
    parser.add_argument('--g_beta2', default=0.999, type=float, help='beta2 for adam optimizer, default 0.999')
    parser.add_argument('--g_weightdecay', default=0, type=float, help='weightdecay for G')

    parser.add_argument('--d_norm_type', default='instance', type=str, help='norm type for D')
    parser.add_argument('--d_lr', default=2e-4, type=float, help='learning rate for optimizer, default 2e-4')
    parser.add_argument('--d_lr_decay_step', default=30, type=int, help='decay step for lr')
    parser.add_argument('--d_lr_decay_gamma', default=0.1, type=float, help='scale ratio for lr')
    parser.add_argument('--d_lr_policy', default='linear', type=str, help='[linear | lambda | step | plateau]')
    parser.add_argument('--d_beta1', default=0.5, type=float, help='beta1 for adam optimizer, default 0.5')
    parser.add_argument('--d_beta2', default=0.999, type=float, help='beta2 for adam optimizer, default 0.999')
    parser.add_argument('--d_weightdecay', default=0, type=float, help='weight decay for D')

    parser.add_argument('--g_input_dim', default=100, type=int, help='dim of input noise, default 100')
    parser.add_argument('--g_output_dim', default=3, type=int, help='dim of output of G, default 3')
    parser.add_argument('--d_input_dim', default=3, type=int, help='channels of input of D, default 3')
    parser.add_argument('--g_net_width', default=64, type=int, help='width for netG')
    parser.add_argument('--d_net_width', default=64, type=int, help='width for netD')

    parser.add_argument('--display_fre', default=400, type=int, help='display frequency, default 400')
    parser.add_argument('--save_fre', default=20, type=int, help='save frequency, default 20')
    parser.add_argument('--visdom_port', default=8891, type=int, help='visdom port, default 8891')
    parser.add_argument('--visdom_env', default='main', type=str, help='visdom env, default main')

    # depends on the network and loss
    parser.add_argument('--loss_choice', default='gan', type=str,
                        help='vanilla| lsgan | wgan | began, default gan')
    parser.add_argument('--gp_choice', default='none', type=str,
                        help='none | wgan-gp | dragan, default None')
    parser.add_argument('--lambdaGP', default=10.0, type=float, help='weight for gradient penalty, default 10')

    # for began
    parser.add_argument('--lambdaK', default=0.001, type=float, help='learning rate for k')
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--lr_update_step', type=int, default=3000)

    # for pix2pix/cyclegan
    parser.add_argument('--train_target_dir', type=str, help='target data dir for cyclegan')
    parser.add_argument('--use_dropout', action='store_true', help='if True, use dropout in G')
    parser.add_argument('--d_num_layers', default=3, type=int, help='layers of PatchGAN for D')
    parser.add_argument('--lambdaI', default=0.5, type=float, help='lambda for Idt')
    # pix2pix
    parser.add_argument('--g_unet_depth', default=8, type=int, help='resnet blocks for G')
    # cyclegan
    parser.add_argument('--g_resnet_blocks', default=9, type=int, help='resnet blocks for G')
    parser.add_argument('--lambdaA', default=10.0, type=float, help='lambda for A')
    parser.add_argument('--lambdaB', default=10.0, type=float, help='lambda for B')

    # for attention-cyclegan
    parser.add_argument('--a_net_width', default=3, type=int, help='width for netA')
    parser.add_argument('--mask_tau', default=0.1, type=float, help='threshold for binary attention mask')
    parser.add_argument('--a_norm_type', default='instance', type=str, help='norm type for netA')
    parser.add_argument('--a_lr', default=2e-4, type=float, help='learning rate for optimizer, default 2e-4')
    parser.add_argument('--a_lr_decay_step', default=30, type=int, help='decay step for lr')
    parser.add_argument('--a_lr_policy', default='lambda', type=str, help='[lambda | step | plateau]')
    parser.add_argument('--a_beta1', default=0.5, type=float, help='beta1 for adam optimizer, default 0.5')
    parser.add_argument('--a_beta2', default=0.999, type=float, help='beta2 for adam optimizer, default 0.999')
    parser.add_argument('--a_weightdecay', default=5e-5, type=float, help='weight decay, default 5e-5')
    parser.add_argument('--train_att_epochs', default=30, type=int, help='epochs to train attention network')
    parser.add_argument('--load_when_switch', action='store_true', help='load state dict when remove IN layers')

    # for openset-faceswap-gan
    parser.add_argument('--ia_output_dim', default=512, type=int, help='dim of ia output, default 512')
    parser.add_argument('--ia_norm_type', default='instance', type=str, help='norm type')
    parser.add_argument('--i_lr', default=2e-4, type=float, help='learning rate for optimizer, default 2e-4')
    parser.add_argument('--i_lr_decay_step', default=30, type=int, help='decay step for lr')
    parser.add_argument('--i_lr_policy', default='lambda', type=str, help='[lambda | step | plateau]')
    parser.add_argument('--i_beta1', default=0.5, type=float, help='beta1 for adam optimizer, default 0.5')
    parser.add_argument('--i_beta2', default=0.999, type=float, help='beta2 for adam optimizer, default 0.999')
    parser.add_argument('--i_weightdecay', default=5e-5, type=float, help='weight decay, default 5e-5')

    parser.add_argument('--lambdaIdt', default=1.0, type=float)
    parser.add_argument('--lambdaCls', default=1.0, type=float)
    parser.add_argument('--lambdaAtt', default=1.0, type=float)
    parser.add_argument('--lambdaGC', default=1.0, type=float)
    parser.add_argument('--lambdaGR', default=1.0, type=float)
    parser.add_argument('--lambdaGD', default=1.0, type=float)
    parser.add_argument('--lambdaGAN', default=1.0, type=float)
    parser.add_argument('--lambdaDiff_alpha', default=0.1, type=float)
    parser.add_argument('--n_classes', default=14, type=int)

    # FOR STARGAN
    parser.add_argument('--lambdaRec', default=10.0, type=float)
    # lambdaCls
    # lambdaGP
    parser.add_argument('--celeba_attr_path', default='', type=str)
    parser.add_argument('--selected_attrs', nargs='+', type=str)
    parser.add_argument('--attr_dim', default=5, type=int)
    parser.add_argument('--d_resnet_blocks', default=6, type=int, help='resnet blocks for D')

    args = parser.parse_args()
    return args


def test_config():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--gpu', default='0', type=str, help='index of gpu to use, default 0')
    parser.add_argument('--test_batch', default=128, type=int, help='test batch size, default 128')
    parser.add_argument('--resize_choice', default=0, type=int,
                        help='0 for ori, 1 for resize, 2 for resize and cent, others for reisze and rand')
    parser.add_argument('--load_size', default=256, type=int, help='load size, default 256')
    parser.add_argument('--fine_size', default=256, type=int, help='fine size, default 256')
    parser.add_argument('--workers', default=4, type=int, help='num of workers to use, default 4')
    parser.add_argument('--flip', default=0, type=int, help='0 for not flip, 1 for hor flip, default 0')
    parser.add_argument('--test_dir', default='', type=str, help='dir for tst data')
    parser.add_argument('--output_dir', default='', type=str, help='output for results')
    parser.add_argument('--save_dir', default='./checkpoints/dcgan/celeba', type=str,
                        help='dir to load model')
    parser.add_argument('--last_epoch', default=0, type=int, help='load model')
    parser.add_argument('--display_fre', default=400, type=int, help='display frequency, default 400')
    parser.add_argument('--visdom_port', default=8891, type=int, help='visdom port')
    parser.add_argument('--visdom_env', default='main', type=str, help='visdom env, default main')
    parser.add_argument('--visdom_win', default='0,1', type=str, help='window id for visdom')

    # depends on the network
    # TODO add test config
    parser.add_argument('--g_resnet_blocks', default=8, type=int, help='resnet blocks for G')
    parser.add_argument('--g_norm_type', default='instance', type=str, help='norm type for g')

    args = parser.parse_args()
    return args
