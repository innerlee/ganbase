import random
import torch
import numpy as np


def random_seed():
    manualSeed = random.randint(1, 10000)
    print(f"Random Seed: {manualSeed}")
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    rng = np.random.RandomState(manualSeed)
