import random
import numpy as np
import torch
import os
import datetime
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt

from utils.arguments import CFGS

# You can ignore the following code.

class Initializer:
    def __init__(self):
        self.__SET_SEED()
        self.__VISUALIZE_TENSOR_SHAPE()
    
    def __SET_SEED(self, seed=1024):
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
        self.testing = cfgs.testing
        self.path = 'log.txt'
        self.checkpoint_path = cfgs.checkpoint_path
        self.result_path = cfgs.result_path

        self.save_path = self.checkpoint_path if not self.testing else self.result_path
        self.file_name = 'log.txt' if not self.testing else cfgs.checkpoint_path.replace('/', '-') + '.txt'

        self.__CLEAR_LOG()
        
    def __CLEAR_LOG(self):
        with open(self.path, 'w+') as f:
            pass

    def SAVE_LOG(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        with open(self.path, 'r') as original:
            with open(self.save_path + self.file_name, 'a') as copy:
                copy.write(f'\n********* {datetime.datetime.now()} *********\n\n')
                for line in original:
                    copy.write(line)        
        self.__CLEAR_LOG()

    def WRITE(self, text):
        with open(self.path, 'a+') as f:
            f.write(text + '\n')

Log = Logger(CFGS)
save_log = Log.SAVE_LOG()
log = Log.WRITE


def LOAD_CHECKPOINT(optimizer, model):
    path = CFGS.checkpoint_path

    checkpoint_path = path + '/checkpoint.tar'
    checkpoint = torch.load(checkpoint_path)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    return start_epoch, optimizer, model
    

def LOAD_WEIGHT(model):
    path = CFGS.checkpoint_path

    checkpoint_path = path + '/checkpoint.tar'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def SAVE_CHECKPOINT(epoch, optimizer, model, path):
    save_dict = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict()
    }
    torch.save(save_dict, os.path.join(path, f'checkpoint.tar'))


# # Start your code.

# def visualize_obj_file(path):

#     # https://zhuanlan.zhihu.com/p/655737746


#     V, F = [], []
#     with open(path) as f:
#         for line in f.readlines():
#             if line.startswith('#'):
#                 continue
#             values = line.split()
#             if not values:
#                 continue
#             if values[0] == 'v':
#                 V.append([float(x) for x in values[1:4]])
#             elif values[0] == 'f':
#                 F.append([int(x.split('/')[0]) for x in values[1:4]])
#     V, F = np.array(V), np.array(F)-1

#     V = (V-(V.max(0)+V.min(0))/2)/max(V.max(0)-V.min(0))

#     fig = plt.figure(figsize=(6,6))
#     T = V[F][...,:2]
#     ax = fig.add_axes([0,0,1,1], xlim=[-1,+1], ylim=[-1,+1],
#                     aspect=1, frameon=False)
#     collection = PolyCollection(T, closed=True, linewidth=0.1,
#                                 facecolor="None", edgecolor="black")
#     ax.add_collection(collection)
#     plt.show()