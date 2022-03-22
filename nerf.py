import math
import time
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt

import loader

from torch.utils.data import DataLoader
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
    def __init__(self, point_dim = 60, dir_dim = 24, depth = 8, width = 256):
        super(Network, self).__init__()
        self.depth = depth
        self.width = width

        # Build layers for space coordinates
        point_layer = torch.nn.Sequential(torch.nn.Linear(point_dim, width), torch.nn.ReLU(True))
        for i in range(1, depth):
            point_layer = torch.nn.Sequential(point_layer, torch.nn.Linear(width, width), torch.nn.ReLU(True))

        self.point_layer = torch.nn.Sequential(point_layer, torch.nn.Linear(width, width))
        self.sigma_layer = torch.nn.Sequential(torch.nn.Linear(width, 1), torch.nn.Sigmoid())

        # Build layers for direction coordinates
        self.color_layer = torch.nn.Sequential(torch.nn.Linear(width + dir_dim, 3), torch.nn.Sigmoid())

    def forward(self, point, dir):
        point_long_vec = torch.flatten(point)
        dir_long_vec = torch.flatten(dir)
    
        point_out = self.point_layer(point_long_vec)
        sigma_out = self.sigma_layer(point_out)
        color_in  = torch.cat((dir_long_vec, point_out), dim = -1)
        color_out = self.color_layer(color_in)
        out = (color_out, sigma_out)

        return out

class Encoder(nn.Module):
    def __init__(self, L_point = 10, L_dir = 4):
        super(Encoder, self).__init__()
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
        dir = dir.reshape(-1, 3)
        dir = dir[0] # transform into a row vector
        for l in range(0, 2 * self.L_dir, 2):
            angle = torch.mul(2 ** l * math.pi, dir)
            gamma_dir[l] = torch.sin(angle)
            gamma_dir[l + 1] = torch.cos(angle)

        gamma_dir = gamma_dir.permute(1, 0)
        gamma = (gamma_point, gamma_dir)

        return gamma

class NeRFModel(nn.Module):
    def __init__(self, num_coarse = 64, num_fine = 128):
        super(NeRFModel, self).__init__()
        self.encoder = Encoder()
        self.network = Network()
        self.num_coarse = num_coarse
        self.num_fine = num_fine

    def net_out(self, t_array, dir_wrd, dir_cam, trans_mat, num_points):
        # Notice: homogeneous coordinates here!
        points_cam = dir_cam.repeat(num_points, 1)
        # Revise Column 2
        points_cam[torch.arange(0, num_points).long(), 2] = t_array.reshape(-1, num_points).double()
        # [[x, y, z, 1], [x, y, z, 1], ...], points in batch
        points_wrd = torch.mm(trans_mat, torch.transpose(points_cam, 0, 1))

        # [[R,G,B], [R,G,B], ...]
        color = torch.rand(num_points, 3).to(device)
        sigma = torch.rand(num_points, 1).to(device)
        points_wrd = torch.transpose(points_wrd[:3, : ], 0, 1)
        dir_wrd = dir_wrd[ :3, : ].reshape(3, -1)
        # Note: Use dataloader?
        for i in range(0, num_points):
            point_enc, dir_enc = self.encoder.forward(points_wrd[i], dir_wrd)
            color[i], sigma[i] = self.network.forward(point_enc, dir_enc)

        # Notice: these two are column vectors: (192, 3), (192, 1)
        return color, sigma

    # Get cdf of coarse sampling, then with its reverse, we use uniform sampling along the horizontal axis
    def resample(self, t_coarse, sigma_coarse):
        # Transform column vector into row one
        sigma_coarse = torch.transpose(sigma_coarse, 0, 1)
        sigma_coarse = sigma_coarse[0]

        cdf = torch.cumsum(sigma_coarse, dim = 0)
        high = max(cdf)
        low = min(cdf)
        delta = t_coarse[1] - t_coarse[0]
        # Slope of cdf is not zero, so its inverse is not infinite
        slope = sigma_coarse[1: ] / delta
        slope_inv = 1.0 / slope
        t_inv = torch.linspace(float(low), float(high), self.num_fine + 2).to(device)
        # Init value
        t_inv = t_inv[1:-1]
        # Row vector
        #print("DEVICE:", cdf.device)
        #exit(0)
        t_fine = (torch.rand(1, self.num_fine))[0].to(device)
        for i in range(0, self.num_fine):
            # select those less than, and keep the last one (max less than)
            sel_le = torch.nonzero(t_inv[i] > cdf)
            sel_le = sel_le[-1]
            # Linear increment
            t_fine[i] = t_coarse[sel_le] + (t_inv[i] - cdf[sel_le]) * slope_inv[sel_le]

        return t_fine

    def color_cum(self, delta, sigma, color):
        sigma_delta = torch.mul(delta, sigma)
        sum_sd = torch.cumsum(sigma_delta, dim = 0)
        T = torch.exp(-sum_sd)
        t_exp = torch.mul(T, 1 - torch.exp(-sigma_delta))
        color = color.transpose(0, 1)
        # Transform into 2 dimensions
        t_exp = t_exp.unsqueeze(0)
        term = torch.mul(color, t_exp)

        return torch.sum(term, dim = 1)

    # Render a ray
    # Local coordinate: [x, y, z] = [right, up, back]
    def render_ray(self, hor, ver, trans_mat, near, far, last = 0.0001):
        trans_mat = trans_mat.to(device)
        d_cam = torch.tensor([hor, ver, 1, 1]).to(torch.float64).to(device)
        d_wrd = torch.mm(trans_mat, d_cam.reshape(4, -1)) # transpose to a column vector

        t_coarse = torch.linspace(near, far, self.num_coarse).to(device)
        color_co, sigma_co = self.net_out(t_coarse, d_wrd, d_cam, trans_mat, self.num_coarse)

        t_fine = self.resample(t_coarse, sigma_co)
        color_fi, sigma_fi = self.net_out(t_fine, d_wrd, d_cam, trans_mat, self.num_fine)

        # Note: here t is for camara or world?
        delta_co = torch.full((1, self.num_coarse), (far - near) / self.num_coarse).to(device)
        # Below: (1, 192), (3, 192), (1, 192)
        t = torch.cat((t_coarse, t_fine)).unsqueeze(dim = 0) # [192] -> [1, 192]
        color = torch.cat((color_co, color_fi)).transpose(0, 1)
        sigma = torch.cat((sigma_co, sigma_fi)).transpose(0, 1)
        # (5, 192)
        sort_bundle = torch.cat((t, color, sigma), dim = 0)
        bundle, _ = torch.sort(sort_bundle, dim = 1) # drop indices here
        t = bundle[0]
        color = bundle[1:4].transpose(0, 1)
        sigma = bundle[4]
        # Add a tiny interval at the tail
        delta = torch.cat((t[1: ] - t[ :-1], torch.tensor([last]).to(device)))

        # Transform into row vectors
        delta_co = delta_co[0]
        sigma_co = sigma_co.transpose(0, 1)[0]

        C_coarse = self.color_cum(delta_co, sigma_co, color_co)
        C_fine = self.color_cum(delta, sigma, color)

        return C_coarse, C_fine

    def ray_loss(self, C_coarse, C_fine, C_true):
        loss_1 = torch.sum(torch.square(C_coarse - C_true))
        loss_2 = torch.sum(torch.square(C_fine - C_true))
        return loss_1 + loss_2

    def forward(self, hor, ver, trans_mat, near, far):
        # In picture: [x, y] = [right, down]
        # Note: ignore batch here
        return self.render_ray(hor, -ver, trans_mat, near, far)

img_dir = "../nerf_llff_data/fern/"
low_res = 8
EPOCH = 10000

def poses_extract(pb_matrix):
    pose = pb_matrix[ :-2].reshape(3, 5)
    c_to_w = torch.tensor(pose[ : , :-1])
    hwf = pose[ : , -1]
    height, width, focal = hwf
    near, far = pb_matrix[-2: ]
    c_to_w = torch.cat((torch.as_tensor(c_to_w).clone(), torch.tensor([[0.0, 0.0, 0.0, 1.0]])), dim = 0)
    return c_to_w, height, width, focal, near, far

def NeRF_trainer():
    model = NeRFModel(num_coarse = 8, num_fine = 16)
    train_dataset = loader.NeRFDataset(root_dir = img_dir + "images_8/", transform = None)
    train_dataloader = DataLoader(dataset = train_dataset, shuffle = True)

    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4, betas = (0.9, 0.999), eps = 1e-7)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.1)

    for epoch in range(EPOCH):
        print("[EPOCH]", epoch)
        loop = tqdm(enumerate(train_dataloader), total = len(train_dataloader))
        for index, (img, poses_bounds) in loop:
            print("[IMG]", index, "[AT TIME]", time.asctime(time.localtime(time.time())))
            poses_bounds = poses_bounds.numpy()
            poses_bounds = poses_bounds[0]
            img = img[0].to(device)
            c_to_w, height, width, focal, near, far = poses_extract(poses_bounds)
            height = int(height) // low_res
            width = int(width) // low_res
            result = torch.rand(height, width, 3)
            for ver in range(0, height):
                print("VER: ", ver, "/", height)
                for hor in range(0, width):
                    # For each ray
                    optimizer.zero_grad()
                    # ver: 378, hor: 504
                    C_true = img[ver, hor]
                    C_coarse, C_fine = model(hor, ver, c_to_w, near, far)
                    loss = model.ray_loss(C_coarse, C_fine, C_true)

                    loss.backward()
                    optimizer.step()
                    result[ver, hor] = C_fine

                if(((ver % 10) == 0) or (ver == height - 1)):
                    plt.imsave("./results/" + str(epoch) + "/" + str(index) + ".jpg", result.detach().numpy())

        scheduler.step()

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
'''
c = torch.tensor(1.8585)
f = torch.tensor([0.0589, 0.1179, 0.1769, 0.2358, 0.2947, 0.3538, 0.4127, 0.4716, 0.5305,
        0.5895, 0.6485, 0.7075, 0.7666, 0.8256, 0.8847, 0.9438, 1.0027, 1.0617,
        1.1206, 1.1796, 1.2387, 1.2976, 1.3565, 1.4155, 1.4743, 1.5333, 1.5923,
        1.6511, 1.7100, 1.7688, 1.8278, 1.8867])
print(torch.nonzero(f < c))
'''
'''
c = torch.tensor([-0.0892, -0.0844, -0.0796, -0.0747, -0.0699, -0.0650, -0.0602, -0.0554,
        -0.0505, -0.0457, -0.0408, -0.0360, -0.0311, -0.0263, -0.0215, -0.0166])
f = torch.tensor([-0.0118, -0.0236, -0.0353, -0.0471, -0.0588, -0.0706, -0.0823, -0.0941])
for k in range(0, 16):
    print(k)
    print(torch.nonzero(c[k] > f)[-1])
'''
#c = torch.rand(3, 64)
#f = torch.rand(1, 64)
#print(c.shape)
#print(f.shape)
#print(torch.mul(c, f))

NeRF_trainer()
