import math
import time
import glob
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import matplotlib.pyplot as plt

import loader

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

MODEL = 0

IMG_DIR = "../nerf_synthetic/lego/" if MODEL > 0 else "../nerf_llff_data/fern/"
MODEL_PATH = "./checkpoint/"
LOW_RES = 1
EPOCH = 10000
BATCH_RAY = 200
LEARNING = 1e-2
LR_GAMMA = 0.5
LR_MILESTONE = [2, 100, 200, 300]
NUM_PIC = 100 if MODEL > 0 else 20
N_COARSE = 64
N_FINE = 128
DATA_TYPE = "sync" if MODEL > 0 else "llff"
STEP = 1


plt.set_cmap("cividis")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

writer = SummaryWriter()
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

def seed_everything(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(624)

class Activation(nn.Module):
    def __init__(self):
        super(Activation, self).__init__()

    def forward(self, x):
        return torch.abs(x)

class Network(nn.Module):
    def __init__(self, point_dim = 60, dir_dim = 24, depth = 8, width = 256, batch_size = 8, layers_skip = [4]):
        super(Network, self).__init__()
        self.depth = depth
        self.width = width
        self.batch_size = batch_size
        self.layers_skip = layers_skip

        # Build layers for space coordinates
        self.point_layer = nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(point_dim, width), torch.nn.ReLU(True))])
        # Choose layers to skip (direct connection of input)
        for i in range(1, depth):
            if i in layers_skip:
                self.point_layer.append(torch.nn.Sequential(torch.nn.Linear(width + point_dim, width), torch.nn.ReLU(True)))
            else:
                self.point_layer.append(torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.ReLU(True)))

        # Note: no need in (0, 1) ?
        self.sigma_layer = torch.nn.Sequential(torch.nn.Linear(width, 1), Activation())
        # Build layers for direction coordinates
        self.point_info = torch.nn.Linear(width, width)
        # get a 128-D feature vector
        self.dir_info = torch.nn.Sequential(torch.nn.Linear(width + dir_dim, width // 2), torch.nn.ReLU(True))
        self.color_layer = torch.nn.Sequential(torch.nn.Linear(width // 2, 3), torch.nn.Sigmoid())

    def forward(self, num_points, point, dir):
        # Shape as (N_batch, N_points, L+L, N_channel)
        point_long_vec = torch.flatten(point, start_dim = 2)
        dir_long_vec = torch.flatten(dir, start_dim = 2)

        point_in = point_long_vec
        for i in range(0, self.depth):
            if i in self.layers_skip:
                point_out = (self.point_layer[i])(torch.cat((point_in, point_long_vec), dim = -1))
            else:
                point_out = (self.point_layer[i])(point_in)

            point_in = point_out

        sigma_out = self.sigma_layer(point_out)
        # encoding point information
        point_info = self.point_info(point_out)
        color_in  = self.dir_info(torch.cat((dir_long_vec, point_info), dim = -1))
        color_out = self.color_layer(color_in)
        # (N_batch, N_point, 3)
        # (N_batch, N_point, 1)
        out = (color_out, sigma_out)

        return out

class Encoder(nn.Module):
    def __init__(self, L_point = 10, L_dir = 4, batch_size = 8):
        super(Encoder, self).__init__()
        self.L_point = L_point
        self.L_dir = L_dir
        self.batch_size = batch_size

    # Point shape as [[[x, y, z], [x, y, z], ... * POINTS], ... * BATCH]
    # (N_batch, N_point, 3)
    def forward(self, num_points, point, dir):
        # num_points: Number of points for each ray
        # x, y, z = point
        # p, q, r = dir

        # Encoder for [x, y, z] and [p, q, r] bundle
        gamma_bundle = torch.rand(1, 1, self.L_point + self.L_dir, 1, 1).to(device)
        gamma_bundle[0, 0, : self.L_point, 0, 0] = torch.linspace(0, self.L_point, self.L_point)
        gamma_bundle[0, 0, self.L_point: , 0, 0] = torch.linspace(0, self.L_dir, self.L_dir)
        # Get 2^l * pi
        gamma_bundle = torch.exp2(gamma_bundle) * math.pi

        # (N_batch, N_points, L, N_channel, 2 (sin, cos))
        gamma_bundle = gamma_bundle.repeat(self.batch_size, num_points, 1, 3, 2)
        # unsqueeze + repeat: (N_batch, N_points, N_channel) -> (N_batch, N_points, L1+L2, N_channel, 2)
        point_bundle = point.unsqueeze(2).unsqueeze(-1).repeat(1, 1, self.L_point, 1, 2)
        dir_bundle = dir.unsqueeze(2).unsqueeze(-1).repeat(1, 1, self.L_dir, 1, 2)
        bundle = torch.cat((point_bundle, dir_bundle), dim = 2)
        gamma_bundle = torch.mul(gamma_bundle, bundle)
        # [[[[sin x, cos x], [sin y, cos y], [sin z, cos z]], ... * L], ... * BATCH]
        # (N_batch, N_points, L, N_channel, 2) -> (N_batch, N_points, L, N_channel) -> (N_batch, N_points, L, N_channel, 1)
        gamma_bundle_sin = torch.sin(gamma_bundle[ : , : , : , : , 0]).unsqueeze(-1)
        gamma_bundle_cos = torch.cos(gamma_bundle[ : , : , : , : , 1]).unsqueeze(-1)
        # (N_batch, N_points, L, N_channel, 1) -> (N_batch, N_points, L, N_channel, 2) -> (N_batch, N_points, N_channel, L, 2) -> (N_batch, N_points, N_channel, L+L)
        gamma_bundle = torch.cat((gamma_bundle_sin, gamma_bundle_cos), dim = -1).permute(0, 1, 3, 2, 4).flatten(start_dim = 3, end_dim = 4)
        # [[[[sin x, cos x, sin 2x, cos 2x, ...], [sin y, cos y, ...], [sin z, ...]], ... * points], ... * batch]

        # (N_batch, N_points, N_channel, L+L)
        gamma_point = gamma_bundle[ : , : , : , : 2 * self.L_point]
        gamma_dir = gamma_bundle[ : , : , : , 2 * self.L_point : ]
        gamma = (gamma_point, gamma_dir)

        return gamma

class NeRFModel(nn.Module):
    def __init__(self, num_coarse = 64, num_fine = 128, batch_ray = 8):
        super(NeRFModel, self).__init__()
        self.encoder = Encoder(batch_size = batch_ray)
        self.network = Network(batch_size = batch_ray)
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.batch_ray = batch_ray

    # Local coordinates: [x, y, z] = [right, up, back]
    def net_out(self, t_array, batch_x, batch_y, trans_mat, K_inv, num_points):
        # Notice: homogeneous coordinates here!
        # t_array: shape as (N_batch, N_points)
        # trans_mat: shape as (N_batch, 4, 4)
        # batch_x: shape as (N_batch)
        # xy_hom: shape as (3, N_batch)
        # Notice: get image inverted!
        xy_hom = torch.cat((
            batch_x.unsqueeze(0),
            batch_y.unsqueeze(0),
            torch.ones(1, self.batch_ray).to(device)), dim = 0)
        # Pixel coordinates -> Camara coordinates
        # (3, N_batch) -> (3, N_batch, N_points) -> (N_batch, N_points, 3)
        # Broadcast multiplication: (N_batch, N_points, 3) * (3, 3) -> (N_batch, N_points, 3)
        points_scale = torch.matmul(xy_hom.unsqueeze(2).repeat(1, 1, num_points).permute(1, 2, 0), K_inv.transpose(0, 1))
        # Scaled by t, use t as z_c
        points_cam = torch.mul(points_scale, t_array.unsqueeze(2).repeat(1, 1, 3))
        # (N_batch, N_points, 4)
        points_cam = torch.cat((points_cam, torch.ones((self.batch_ray, num_points, 1)).to(device)), dim = 2)
        # Notice: dir vector is computed the same way as points, but we use `norm` here!
        # (N_batch, N_points, 3)
        dir_cam = functional.normalize(points_scale, p = 2.0, dim = 2)
        # [[x, y, z, 1], [x, y, z, 1], ...], points in batch
        # (N_batch, 4, 4) -> (N_batch, N_points, 4, 4)
        batch_mat = trans_mat.unsqueeze(1).repeat(1, num_points, 1, 1)
        # (N_batch, N_points, 4) -> (N_batch, N_points, 4, 1)
        # (N_batch, N_points, 4, 4) * (N_batch, N_points, 4, 1) -> (N_batch, N_points, 4, 1) -> (N_batch, N_points, 4)
        points_wrd = torch.matmul(batch_mat, points_cam.unsqueeze(3)).squeeze()
        # (N_batch, N_points, 3)
        # Only rotation for vector
        dir_wrd = torch.matmul(batch_mat[ : , : , : 3, : 3], dir_cam.unsqueeze(3)).squeeze()

        # [[[R,G,B], [R,G,B], ... * points], ... * batch]
        # (N_batch, N_points, 3)
        points_wrd = points_wrd[ : , : , :3]

        point_enc, dir_enc = self.encoder.forward(num_points, points_wrd, dir_wrd)
        color, sigma = self.network.forward(num_points, point_enc, dir_enc)

        # output shape: (N_batch, N_points, channel=3/1)
        return color, sigma

    # Get cdf of coarse sampling, then with its reverse, we use uniform sampling along the horizontal axis
    def resample(self, t_coarse, sigma_coarse):
        # t_coarse: (N_batch, N_c)
        # sigma_coarse: (N_batch, N_c, 1) -> (N_batch, N_c)
        sigma_coarse = sigma_coarse.squeeze()

        # (N_batch, N_c)
        cdf = torch.cumsum(sigma_coarse, dim = 1).contiguous()
        # drop indices
        # shape: (N_batch)
        high, _ = torch.max(cdf, dim = 1)
        low, _ = torch.min(cdf, dim = 1)
        delta = t_coarse[0, 1] - t_coarse[0, 0]
        EPSILON = 1e-5
        # Slope of cdf is not zero, so its inverse is not infinite
        # cdf - cdf = sigma
        # Add epsilon to avoid zero-division
        slope_inv = delta / (sigma_coarse[ : , 1: ] + EPSILON)
        high = high.detach().cpu().numpy()
        low = low.detach().cpu().numpy()
        # (N_fine+2, N_batch)
        t_inv = np.linspace(tuple(low), tuple(high), self.num_fine + 2)
        # Init value, drop start and end
        # (N_batch, N_fine)
        t_inv = torch.tensor(t_inv[1 : -1]).to(device).transpose(0, 1).contiguous()
        # indices of t_inv when inserted in cdf
        index_fine = torch.searchsorted(cdf, t_inv) - 1

        # Add an extra column to fit the function, but we will not use it then
        if len(torch.nonzero(index_fine > self.num_fine - 1)) > 0 or len(torch.nonzero(index_fine < 0)) > 0:
            print("---------------------------Index: 1--------------------------------")
            print("----------------------------SIGMA----------------------------------")
            print(sigma_coarse)
            np.savetxt("SC" + str(MODEL) + ".txt", np.array(sigma_coarse.detach().cpu()))
            print("----------------------------T_INV----------------------------------")
            print(t_inv)
            np.savetxt("TI" + str(MODEL) + ".txt", np.array(t_inv.detach().cpu()))
            print("-----------------------------INDEX---------------------------------")
            print(index_fine)
            np.savetxt("IF" + str(MODEL) + ".txt", np.array(index_fine.detach().cpu()))
            print("-----------------------------T_C-----------------------------------")
            print(t_coarse)
            np.savetxt("TC" + str(MODEL) + ".txt", np.array(t_coarse.detach().cpu()))
            print("--------------------------SLOPE_INV--------------------------------")
            print(slope_inv)
            np.savetxt("SI" + str(MODEL) + ".txt", np.array(slope_inv.detach().cpu()))
            print("------------------------------CDF----------------------------------")
            print(cdf)
            np.savetxt("CD" + str(MODEL) + ".txt", np.array(cdf.detach().cpu()))
            exit(0)

        lower_t = torch.gather(t_coarse, dim = 1, index = index_fine)
        lower_cdf = torch.gather(cdf, dim = 1, index = index_fine)
        temp = torch.cat((slope_inv, torch.zeros(self.batch_ray, 1).to(device)), dim = 1)
        lower_slope = torch.gather(temp, dim = 1, index = index_fine)
        t_fine = lower_t + (t_inv - lower_cdf) * lower_slope

        return t_fine

    def color_cum(self, delta, sigma, color):
        # delta: (N_batch, N_points)
        # sigma: (N_batch, N_points)
        # color: (N_batch, N_points, 3)
        sigma_delta = torch.mul(delta, sigma)
        sum_sd = torch.cumsum(sigma_delta, dim = 1)
        T = torch.exp(-sum_sd)
        # (N_batch, N_points) -> (N_batch, N_points, 1)
        t_exp = torch.mul(T, 1 - torch.exp(-sigma_delta)).unsqueeze(2)
        term = torch.mul(color, t_exp)
        result = torch.sum(term, dim = 1)

        return result

    # Render a ray batch (drop last batch)
    # Local coordinate: [x, y, z] = [right, up, back]
    # Notice: some redundant calculation here!
    def render_rays(self, batch_hor, batch_ver, trans_mat, K_inv, near, far, last = 0.0001):
        # Shape as (N_batch, N_c)
        t_coarse = torch.tensor(np.linspace(tuple(near), tuple(far), self.num_coarse)).transpose(0, 1).to(device)
        color_co, sigma_co = self.net_out(t_coarse, batch_hor, batch_ver, trans_mat, K_inv, self.num_coarse)

        # Shape as (N_batch, N_f)
        t_fine = self.resample(t_coarse, sigma_co)
        color_fi, sigma_fi = self.net_out(t_fine, batch_hor, batch_ver, trans_mat, K_inv, self.num_fine)

        # Note: here t is for camara or world?
        # far, near: (N_batch)
        # (N_batch, N_c)
        delta_co = ((far - near) / self.num_coarse).unsqueeze(1).repeat(1, self.num_coarse).to(device)
        # (N_batch, N_c) + (N_batch, N_f) -> (N_batch, N_c+N_f) -> (N_batch, N, 1)
        t = torch.cat((t_coarse, t_fine), dim = 1).unsqueeze(2)
        # (N_batch, N_point, N_channel), N_point = N_c + N_f
        color = torch.cat((color_co, color_fi), dim = 1)
        sigma = torch.cat((sigma_co, sigma_fi), dim = 1)
        # (N_batch, N_c+N_f, 5)
        sort_bundle = torch.cat((t, color, sigma), dim = 2)
        #print(sort_bundle)
        bundle, _ = torch.sort(sort_bundle, dim = 1) # drop indices here
        #print(bundle)

        t = bundle[ : , : , 0] # (N_batch, N_points)
        color = bundle[ : , : , 1:4] # (N_batch, N_points, 3)
        sigma = bundle[ : , : , 4] # (N_batch, N_points)

        # Add a tiny interval at the tail
        delta = torch.cat((t[ : , 1: ] - t[ : , :-1], torch.full((self.batch_ray, 1), last).to(device)), dim = 1)

        # (N_batch, 3)
        C_coarse = self.color_cum(delta_co, sigma_co[ : , : , 0], color_co)
        C_fine = self.color_cum(delta, sigma, color)

        return C_coarse, C_fine

    def ray_loss(self, C_coarse, C_fine, C_true):
        # (N_batch, 3)
        # sum along both dimensions
        loss_1 = torch.sum(torch.square(C_coarse - C_true))
        loss_2 = torch.sum(torch.square(C_fine - C_true))

        return loss_1 + loss_2

    def forward(self, batch_hor, batch_ver, trans_mat, K_inv, near, far):
        # In picture: [x, y] = [right, down]
        # K: intrinsic matrix (K_inv)
        #trans_mat = trans_mat.to(device)
        #K_inv = K_inv.to(device)
        return self.render_rays(batch_hor, batch_ver, trans_mat, K_inv, near, far)


def poses_extract(pb_matrix):
    # pb shape: [N_batch, 17]
    # [N_batch, 3, 5]
    pose = pb_matrix[ : , :-2].reshape(-1, 3, 5)
    # Notice: near, far are not the same among pixels
    near = pb_matrix[ : , -2]
    far = pb_matrix[ : , -1]
    # [N_batch, 3, 4] + [N_batch, 1, 4] -> [N_batch, 4, 4]
    c_to_w = torch.cat((pose[ : , : , :-1], torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(BATCH_RAY, 1, 1)), dim = 1)
    # Note: suppose they are the same
    # Notice: for focal: suppose unit length is 1 pixel
    height = pose[0, 0, -1]
    width = pose[0, 1, -1]
    focal = pose[0, 2, -1]
    return c_to_w, height, width, focal, near, far

def NeRF_trainer():
    model = NeRFModel(num_coarse = N_COARSE, num_fine = N_FINE, batch_ray = BATCH_RAY).to(device)
    train_dataset = loader.NeRFDataset(root_dir = IMG_DIR, low_res = LOW_RES, transform = None, type = DATA_TYPE)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = BATCH_RAY, shuffle = True, num_workers = 1)

    optimizer = torch.optim.Adam(model.network.parameters(), lr = LEARNING, betas = (0.9, 0.999), eps = 1e-7)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = LR_MILESTONE, gamma = LR_GAMMA)

    # Check existing checkpoint
    ck_list = glob.glob(MODEL_PATH + "*.pkl")
    last_epoch = -1
    for file in ck_list:
        ck = file.split("\\")[-1]
        ep = int(ck[ : -4])
        last_epoch = max(last_epoch, ep)

    if ck_list:
        print("Last epoch:", last_epoch)
        model = torch.load(MODEL_PATH + str(last_epoch) + ".pkl")

    iter = 0
    for epoch in range(last_epoch + 1, EPOCH, 1):
        print("[EPOCH]", epoch)
        for index, (row, column, pix_val, poses_bound) in enumerate(train_dataloader):
            #print("[IMG]", index, "[AT TIME]", time.asctime(time.localtime(time.time())))
            # [N_batch, 17]
            poses_bound = poses_bound.to(torch.float)
            c_to_w, height, width, focal, near, far = poses_extract(poses_bound)
            height = int(height) // LOW_RES
            width = int(width) // LOW_RES
            # inverse of intrinsic matrix
            K_inv = torch.tensor([[1.0 / focal, 0.0, -0.5 * width / focal], [0.0, -1.0 / focal, 0.5 * height / focal], [0.0, 0.0, -1.0]]).to(torch.float).to(device)
            #result = torch.full((height, width, 3), 1.0)

            # Note: here spatial correlation is dropped
            # [N_batch]
            batch_y = row.to(device)
            batch_x = column.to(device)
            # [N_batch, N_channel]
            C_true = pix_val.to(device)
            # [N_batch, 4, 4]
            c_to_w = c_to_w.to(device)

            avg_loss = 0.0

            # For ray batch
            optimizer.zero_grad()
            model.train()
            # ver: 3024, hor: 4032
            C_coarse, C_fine = model(batch_x, batch_y, c_to_w, K_inv, near, far)
            #print(C_coarse)
            loss = model.ray_loss(C_coarse, C_fine, C_true)
            avg_loss += float(loss)

            loss.backward()
            optimizer.step()
            #scheduler.step()
            #result[batch_y, batch_x] = C_fine.cpu()

            # Use tensorboard to record
            writer.add_scalar("loss/train", loss, iter)
            writer.add_scalar("lr/train", optimizer.state_dict()['param_groups'][0]['lr'], iter)
            writer.flush()

            iter += 1

            if (iter % STEP) == 0:
                print("[BATCH]", index, " [LOSS] %.4f "%float(loss),
                      "[T] (%.4f"%float(C_true[0][0]),"%.4f"%float(C_true[0][1]),"%.4f)"%float(C_true[0][2]),
                      "[F] (%.4f"%float(C_fine[0][0]),"%.4f"%float(C_fine[0][1]),"%.4f)"%float(C_fine[0][2]))

            if (iter % (height * width)) == 0:
                print("[AVG] %.4f"%avg_loss)
                avg_loss = 0.0


        #torch.save(model, MODEL_PATH + str(epoch) + ".pkl")


def test():
    m = torch.tensor([[0, 0, 1, 61, 62, 62],
                      [0, 0, 1, 61, 62, 62],
                      [0, 0, 1, 61, 62, 62],
                      [0, 0, 1, 61, 62, 62]])
    n = torch.tensor([[ 2,  7,  8],
                      [10, 11, 12]])
    print(len(torch.nonzero(m[ : , -1] > 127)))


if __name__ == "__main__":
    NeRF_trainer()
    #test()
