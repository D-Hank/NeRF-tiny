import numpy as np
import os
import json
import torch

from torch.utils.data import Dataset
from PIL import Image

NEAR_FACTOR = 2.0
FAR_FACTOR = 6.0

def create_npy(root_dir, mode):
    with open(root_dir + "transforms_" + mode + ".json") as json_file:
        jf = json.load(json_file)

    angle = jf['camera_angle_x']
    frame = jf['frames']
    pic_num = len(frame)

    # read one img to see
    img_0 = Image.open(root_dir + frame[0]['file_path'][2: ] + ".png")
    width, height = img_0.size
    focal = 0.5 * width / np.tan(0.5 * angle)
    near = NEAR_FACTOR
    far = FAR_FACTOR

    img_0 = img_0.convert("RGB").load()
    poses_bounds = np.zeros((pic_num, 17))

    for i in range(pic_num):
        matrix = np.array(frame[i]['transform_matrix'])
        # Column vector
        poses_bound = np.concatenate((np.concatenate((matrix[ :3, :4], np.array([[height], [width], [focal]])), axis = 1).flatten(), np.array([near, far])), axis = 0)
        poses_bounds[i] = poses_bound

    np.save(root_dir + mode + ".npy", poses_bounds)

def convert_npy(root_dir):
    src_trans = np.load(root_dir + "poses_bounds.npy")
    dest_trans = np.zeros_like(src_trans)
    len, _ = src_trans.shape
    print(src_trans[0])
    for i in range(len):
        mat = src_trans[i]
        pose = mat[ :-2].reshape(3, 5)
        near_far = mat[-2: ]
        c_to_w = pose[ : , :4]
        hwf = pose[ : , 4]
        new_ctw = np.concatenate((c_to_w[ : , 1], -c_to_w[ : , 0], c_to_w[ : , 2]), axis = 0)
        new_pose = np.concatenate((new_ctw.reshape(3, 3).transpose(), c_to_w[ : , 3].reshape(3, 1), hwf.reshape(3, 1)), axis = 1).flatten()
        dest_trans[i] = np.concatenate((new_pose, near_far), axis = 0)

    np.save(root_dir + "new.npy", dest_trans)

def data_preprocess(root_dir, type, mode):
    if type == "llff":
        convert_npy(root_dir)
    else:
        create_npy(root_dir, mode)

class NeRFDataset(Dataset):
    # REFERENCE: https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
    def get_img(self, img_path):
        image = Image.open(img_path)
        image.load()

        if self.type == "sync":
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask = image.split()[3]) # 3 is the alpha channel

            image = background

        return np.array(image) / 255.0

    def get_all_pix(self):
        self.height = int(self.poses_bounds[0][4])
        self.width = int(self.poses_bounds[0][9])
        self.focal = self.poses_bounds[0][14]
        self.pic_size = self.height * self.width
        self.num_pix = self.pic_size * self.pic_num

        all_img = torch.zeros(self.pic_num, self.height, self.width, 3)
        for i in range(0, self.pic_num):
            all_img[i] = torch.tensor(self.get_img(self.file_list[i]))

        # (N_pic, height, width, 3) -> (N_pic * H * W, 3)
        # with: W -> H -> N_pic
        self.all_pix = torch.flatten(all_img, start_dim = 0, end_dim = 2)

    def __init__(self, root_dir, low_res = 8, transform = None, type = "sync", mode = "train"):
        self.root_dir = root_dir
        self.low_res = low_res
        self.transform = transform
        self.file_list = []
        self.pic_num = 0
        self.type = type

        trans_path = root_dir + ("new.npy" if type == "llff" else (mode + ".npy"))

        if os.path.isfile(trans_path) == False:
            data_preprocess(root_dir, type, mode)

        self.poses_bounds = np.load(trans_path)

        img_dir = root_dir + ("images/" if type == "llff" else (mode + "/"))

        for file in os.listdir(img_dir):
            self.file_list.append(os.path.join(img_dir, file))
            self.pic_num += 1

        # Notice: sorting is necessary here!
        self.file_list.sort(key = lambda name: int(name.split("_")[-1][:-4]))

        self.get_all_pix()

    def __len__(self):
        return self.num_pix

    def __getitem__(self, idx):
        # idx: this pixel is which pixel among all pixels
        pixel = self.all_pix[idx]
        pix_val = pixel[0 : 3]
        # belongs to which pic
        pic = idx // self.pic_size
        # which pix in this pic
        id_in_pic = idx % self.pic_size
        # which row
        row = id_in_pic // self.width
        # which column
        column = id_in_pic % self.width
        poses_bound = self.poses_bounds[pic]

        return row, column, pix_val, poses_bound, pic
