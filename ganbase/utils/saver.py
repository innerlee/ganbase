import os
from collections import deque
import torch


class Saver(object):
    """
    A wrapper around `torch.save`.

    args
    `slot`: how many models to save for each iteration
    `keepnum`: maximum number of files saved, default 3.
        0 for no save,
        -1 for saving all.
    """

    def __init__(self, slot, keepnum=3):
        self.snap = deque([])
        self.slot = slot
        self.keepnum = keepnum

    def save(self, obj, f):
        torch.save(obj, f)
        self.snap.append(f)
        if len(self.snap) == (self.keepnum + 1) * self.slot:
            for _ in range(self.slot):
                os.remove(self.snap.popleft())

    def load(self, obj, f):
        obj.load_state_dict(torch.load(f))