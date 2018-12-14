import os
import torch


def visible_gpu(gpus):
    """
        set visible gpu.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)
    torch.backends.cudnn.benchmark = True

def occupy_gpu():
    """
        make program appear on nvidia-smi.
    """
    torch.zeros(1).cuda()
