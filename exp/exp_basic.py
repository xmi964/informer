import os
import torch
import numpy as np

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None
    
    def _acquire_device(self):
        if self.args.use_gpu:
            try:
                device = torch.device('cuda:0')
                _ = torch.tensor([0]).to(device)  # 测试CUDA是否真正可用
                print('Use GPU:', device)
                return device
            except:
                print('Warning: CUDA not available, use CPU instead')
        return torch.device('cpu')  # 强制回退到CPU

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
    