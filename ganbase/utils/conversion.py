import numpy as np

def bce2prob(loss, target):
    if target == 1:
        return np.exp(-loss)
    elif target == 0:
        return 1 - np.exp(-loss)
    else:
        raise ValueError('target should be 0 or 1')
