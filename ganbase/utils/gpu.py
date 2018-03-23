import os


def visible_gpu(gpus):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)
