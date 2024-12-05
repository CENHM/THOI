import random
import numpy as np
import torch
import os
import datetime
import torch.optim as optim
import torch.nn as nn
# from matplotlib.collections import PolyCollection
# import matplotlib.pyplot as plt

from utils.arguments import CFGS

# You can ignore the following code.

class Initializer:
    def __init__(self):
        self.__SET_SEED()
        self.__VISUALIZE_TENSOR_SHAPE()
    
    def __SET_SEED(self, seed=1025):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For visualizing the shape of tensor on VSCode
    def __VISUALIZE_TENSOR_SHAPE(self):
        normal_repr = torch.Tensor.__repr__
        torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"

INITIALIZER = Initializer()


class Logger:
    def __init__(self, cfgs):
        self.testing = cfgs.inferencing
        self.path = 'log.txt'
        self.checkpoint_dir = cfgs.checkpoint_dir
        self.result_path = cfgs.result_path

        self.save_path = self.checkpoint_dir if not self.testing else self.result_path
        self.file_name = 'log.txt' if not self.testing else cfgs.checkpoint_dir.replace('/', '-') + '.txt'

        self.__INITIALIZE()

    def __INITIALIZE(self):
        with open(self.path, 'w+') as logf:
            logf.write(f'\n********* {datetime.datetime.now()} *********\n\n')      
        
    def __CLEAR_LOG(self):
        with open(self.path, 'w+') as f:
            pass

    def SAVE_LOG(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        with open(self.path, 'r') as original:
            with open(f'{self.save_path}/{self.file_name}', 'a') as copy:
                copy.write(f'\n********* {datetime.datetime.now()} *********\n\n')
                for line in original:
                    copy.write(line)        
        self.__CLEAR_LOG()

    def WRITE(self, text):
        with open(self.path, 'a+') as f:
            f.write(text + '\n')

Log = Logger(CFGS)
SAVE_LOG = Log.SAVE_LOG
log = Log.WRITE


def LOAD_CHECKPOINT(optims, models):
    log(f'Resume checkpoint from {CFGS.checkpoint_dir}')

    checkpoint_path = f'{CFGS.checkpoint_dir}/checkpoint.tar'
    save_dict = torch.load(checkpoint_path, weights_only=True)

    start_epoch = save_dict['epoch']

    for model_name, model in models.items():
        model.load_state_dict(save_dict['models'][model_name])
    for optim_name, optim in optims.items():  
        optim.load_state_dict(save_dict['optims'][optim_name])  
    
    return start_epoch, optims, models
    

def LOAD_WEIGHT(models: dict):
    log(f'Load trained model from {CFGS.checkpoint_dir}')

    checkpoint_path = f'{CFGS.checkpoint_dir}/checkpoint.tar'
    save_dict = torch.load(checkpoint_path, weights_only=True)

    for model_name, model in models.items():
        model.load_state_dict(save_dict['models'][model_name])
    return models


def SAVE_CHECKPOINT(epoch, optims: dict, models: dict):
    save_dict = {
        'epoch': epoch,
        'models': {},
        'optims': {},
    }
    for model_name, model in models.items():
        save_dict['models'][model_name] = model.state_dict()
    for optim_name, optim in optims.items():
        save_dict['optims'][optim_name] = optim.state_dict()

    torch.save(save_dict, f'{CFGS.checkpoint_dir}/checkpoint.tar')


def ADD_OPTIMIZERS(models: dict):
    optims = {}
    for model_name, model in models.items():
        optims[f'optim_{model_name}'] = optim.Adam(
            model.parameters(), lr=CFGS.learning_rate, weight_decay=CFGS.weight_decay)
    return optims


def OPTIMIZER_STEP(optims: dict):
    for optim_name, optim in optims.items():
        optim.step()
    return optims


def MODELS_SET_MODE(models: dict, train=True):
    for model_name, model in models.items():
        if train:
            model.train()
        else:
            model.eval()
    return models


def MODELS_SET_ZERO_GRAD(models: dict):
    """ Set models to zero gradient during training procedure.
    Params:
        - models: dict(nn.Module)
    Return:
        - models: dict(nn.Module)
    """
    for model_name, model in models.items():
        model.zero_grad()
    return models


def EXAM_GRAD(model: nn.Module):
    for name, parms in model.named_parameters():	
        print('-->name:', name)
        print('-->grad_requirs:',parms.requires_grad)
        print('-->grad_value:', parms.grad)