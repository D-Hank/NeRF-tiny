from turtle import back
from matplotlib.pyplot import axis
import numpy as np
import os
import json
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

NEAR_FACTOR = 2.0
FAR_FACTOR = 6.0

def create_npy(root_dir):
    with open(root_dir + "transforms_train.json") as json_file:
        jf = json.load(json_file)

    angle = jf['camera_angle_x']
    frame = jf['frames']
    pic_num = len(frame)

    # read one img to see
    img_0 = Image.open(root_dir + frame[0]['file_path'][2: ] + ".png")
    width, height = img_0.size
    focal = 0.5 * width * np.tan(0.5 * angle)
    near = NEAR_FACTOR
    far = FAR_FACTOR

    img_0 = img_0.convert("RGB").load()
    poses_bounds = np.zeros((pic_num, 17))

    for i in range(pic_num):
        matrix = np.array(frame[i]['transform_matrix'])
        # Column vector
        poses_bound = np.concatenate((np.concatenate((matrix[ :3, :4], np.array([[height], [width], [focal]])), axis = 1).flatten(), np.array([near, far])), axis = 0)
        poses_bounds[i] = poses_bound

    np.save(root_dir + "new.npy", poses_bounds)

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


class NeRFDataset(Dataset):
    def __init__(self, root_dir, low_res = 8, transform = None, type = "sync"):
        self.root_dir = root_dir
        self.low_res = low_res
        self.transform = transform
        self.file_list = []
        self.pic_num = 0
        self.type = type

        if type == "llff":
            if os.path.isfile(root_dir + "new.npy") == False:
                convert_npy(root_dir)

            if low_res == 8:
                img_dir = root_dir + "images_8/"
            else:
                img_dir = root_dir + "images/"

        if type =="sync":
            if os.path.isfile(root_dir + "new.npy") == False:
                create_npy(root_dir)

            img_dir = root_dir + "train/"

        self.poses_bounds = np.load(root_dir + "new.npy")

        for file in os.listdir(img_dir):
            self.file_list.append(os.path.join(img_dir, file))
            self.pic_num += 1

    # REFERENCE: https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
    def get_img(self, img_path):
        image = Image.open(img_path)
        image.load()

        if self.type == "llff":
            return np.array(image) / 255.0

        else:
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask = image.split()[3]) # 3 is the alpha channel

            return np.array(background) / 255.0

    def __len__(self):
        return self.pic_num

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = self.get_img(img_path)
        poses_bound = self.poses_bounds[idx]

        #print(poses_bounds[idx])
        return img, poses_bound

'''
png = Image.open("../nerf_synthetic/lego/train/r_0.png")
png.load()
background = Image.new('RGB', png.size, (255, 255, 255))
background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
background.save('test.jpg', 'JPEG', quality=80)
'''
