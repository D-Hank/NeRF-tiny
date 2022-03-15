import math
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt

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
        self.color_layer = torch.nn.Sequential(torch.nn.Linear(width + dir_dim, 3), torch.nn.Sigmoid())

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
        gamma = (gamma_point, gamma_dir)

        return gamma

class NeRFModel():
    def __init__(self, num_coarse = 64, num_fine = 128):
        self.encoder = Encoder()
        self.network = Network()
        self.num_coarse = num_coarse
        self.num_fine = num_fine

    def net_out(self, t_array, dir_wrd, dir_cam, trans_mat, num_points):
        # Notice: homogeneous coordinates here!
        points_cam = dir_cam.repeat(num_points, 1)
        points_cam[:, 2] = t_array
        # [[x, y, z, 1], [x, y, z, 1], ...], points in batch
        points_wrd = torch.mm(trans_mat, torch.transpose(points_cam))

        # [[R,G,B], [R,G,B], ...]
        color = torch.rand(num_points, 3)
        sigma = torch.rand(num_points, 1)
        points_wrd = torch.transpose(points_wrd[:3, : ])
        dir_wrd = torch.transpose(dir_wrd[ :3, : ])
        # Note: Use dataloader?
        for i in range(0, num_points):
            point_enc, dir_enc = self.encoder.forward(points_wrd[i], dir_wrd)
            color[i], sigma[i] = self.network.forward(point_enc, dir_enc)

        return color, sigma

    # Get cdf of coarse sampling, then with its reverse, we use uniform sampling along the horizontal axis
    def resample(self, t_coarse, sigma_coarse):
        cdf = torch.cumsum(sigma_coarse, dim = 0)
        high = max(cdf)
        low = min(cdf)
        delta = t_coarse[1] - t_coarse[0]
        # Slope of cdf is not zero, so its inverse is not infinite
        slope = sigma_coarse[1: ] / delta
        slope_inv = 1.0 / slope
        t_inv = torch.linspace(low, high, self.num_fine + 2)
        # Init value
        #print(t_inv)
        t_inv = t_inv[1:-1]
        t_fine = torch.rand(self.num_fine, 1)
        for i in range(0, self.num_fine):
            # select those less than
            sel_le = torch.nonzero(t_inv[i] > cdf)
            sel_le = sel_le[-1]
            # Linear increment
            t_fine[i] = t_coarse[sel_le] + (t_inv[i] - cdf[sel_le]) * slope_inv[sel_le]

        #print(t_coarse)
        #print(t_fine)
        return t_fine

    def color_cum(self, delta, sigma, color):
        sigma_delta = torch.mul(delta, sigma)
        sum_sd = torch.cumsum(sigma_delta)
        T = torch.exp(-sum_sd)
        t_exp = torch.mul(T, 1 - torch.exp(-sigma_delta))
        term = torch.mul(color, t_exp)

        return torch.sum(term, dim = 1)

    # Render a ray
    # Local coordinate: [x, y, z] = [right, up, back]
    def render_ray(self, hor, ver, trans_mat, near, far, last = 0.0001):
        d_cam = torch.tensor([hor, ver, 1, 1])
        d_wrd = torch.mm(trans_mat, torch.transpose(d_cam))

        t_coarse = torch.linspace(near, far, self.num_coarse)
        color_co, sigma_co = self.net_out(t_coarse, d_wrd, d_cam, trans_mat, self.num_coarse)

        t_fine = self.resample(t_coarse, sigma_co)
        color_fi, sigma_fi = self.net_out(t_fine, d_wrd, d_cam, trans_mat, self.num_fine)

        # Note: here t is for camara or world?
        delta_co = torch.full((1, self.num_coarse), (far - near) / self.num_coarse)
        t = torch.cat(t_coarse, t_fine)
        color = torch.cat(color_co, color_fi)
        sigma = torch.cat(sigma_co, sigma_fi)
        sort_bundle = torch.vstack(t, torch.transpose(color), sigma)
        bundle = torch.sort(sort_bundle, dim = 1)
        t = bundle[0]
        color = bundle[1:4]
        sigma = bundle[4]
        # Add a tiny interval at the tail
        delta = torch.cat(t[1: ] - t[ :-1], last)

        C_coarse = self.color_cum(delta_co, sigma_co, color_co)
        C_fine = self.color_cum(delta, sigma, color)

        return C_coarse, C_fine


'''
#t_c = torch.tensor([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
#s_c = torch.tensor([0.1, 0.15, 0.23, 0.3, 0.5, 0.8, 1.2, 0.9, 0.6, 0.38, 0.2])
t_c = torch.tensor([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
s_c = torch.tensor([1.2, 0.9, 0.8, 0.5, 0.3, 0.6,  0.38, 0.1, 0.15, 0.23, 0.2])
nerf = NeRFModel(num_fine = 10)
nerf.resample(t_c, s_c)
'''

#c = torch.tensor([[2, 3, 4], [4, 5, 6]])
#f = torch.tensor([2, 1, 2])
#print(torch.mul(c, f))
