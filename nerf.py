import math
from turtle import color, forward
from unittest import skip
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from tqdm.notebook import tqdm

plt.set_cmap("cividis")

DATASET_PATH = "C:\\Users\\Admin\\Desktop\\NeRF\\nerf_llff_data\\fern"
CHECKPOINT_PATH = "C:\\Users\\Admin\\Desktop\\NeRF\\nerf_reproduce\\checkpoint"

pl.seed_everything(624)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

class Network(nn.Module):
    def __init__(self, point_dim = 60, dir_dim = 16, depth = 24, width = 256):
        super(Network, self).__init__()
        self.depth = depth
        self.width = width

        # Build layers for space coordinates
        point_layer = torch.nn.Sequential(torch.nn.Linear(point_dim, width), torch.nn.ReLU(True))
        for i in range(1, depth):
            point_layer = torch.nn.Sequential(point_layer, torch.nn.Linear(width, width), torch.nn.ReLU(True))

        self.point_layer = torch.nn.Sequential(point_layer, torch.nn.Linear(width, width))
        self.sigma_layer = torch.nn.Linear(width, 1)

        # Build layers for direction coordinates
        self.color_layer = torch.nn.Sequential(torch.nn.Linear(width + point_dim, 3), torch.nn.Sigmoid())

    def forward(self, point, dir):
        point_out = self.point_layer(point)
        sigma_out = self.sigma_layer(point_out)
        color_in  = torch.cat([dir, point_out], dim = -1)
        color_out = self.color_layer(color_in)
        out = (color_out, sigma_out)

        return out

class Encoder():
    def __init__(self, L_point = 10, L_dir = 4):
        self.L_point = L_point
        self.L_dir = L_dir

    def forward(self, point, dir):
        #x, y, z = point
        #p, q, r = dir

        # Encoder for [x, y, z]
        gamma_point = torch.rand(2 * self.L_point, 3)
        # [[sin x, sin y, sin z], [cos x, cos y, cos z], ...]
        for l in range(0, 2 * self.L_point, 2):
            angle = torch.mul(2 ** l * math.pi, point)
            gamma_point[l] = torch.sin(angle)
            gamma_point[l + 1] = torch.cos(angle)

        gamma_point = gamma_point.permute(1, 0)
        # [[sin x, cos x, sin 2x, cos 2x, ...], [sin y, cos y, ...], [sin z, ...]]

        # Encoder for [p, q, r]
        gamma_dir = torch.rand(2 * self.L_dir, 3)
        for l in range(0, 2 * self.L_dir, 2):
            angle = torch.mul(2 ** l * math.pi, dir)
            gamma_point[l] = torch.sin(angle)
            gamma_point[l + 1] = torch.cos(angle)

        gamma_dir = gamma_dir.permute(1, 0)
        gamma = torch.cat(gamma_point, gamma_dir)

        return gamma

class NeRFModel():
    def __init__(self):
        self.encoder = Encoder()
        self.network = Network()


#a = [[1, 0.1, 0.01], [-1, -0.1, -0.01], [2, 0.2, 0.02], [-2, -0.2, -0.02], [3, 0.3, 0.03], [-3, -0.3, -0.03]]
#a = torch.tensor(a)
#print(a.permute(1, 0))

