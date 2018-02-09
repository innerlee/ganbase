import os
import pytest
import torch
import torchvision
import ganbase as gb


class TestTorch(object):

    def test_torch(self):
        assert torch.__version__ == '0.3.0.post4'

    def test_torchvision(self):
        assert torchvision.__version__ == '0.2.0'
