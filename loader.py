import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

img_dir = "../nerf_llff_data/fern/"
low_res = 8

poses_bounds = np.load(img_dir + "poses_bounds.npy")
#np.savetxt("poses_bounds.txt", poses_bounds)
#simplices = np.load(img_dir + "simplices.npy")

#poses = poses_bounds[:, :-2].reshape(-1, 3, 5)
#print(poses_bounds)
#print(poses.shape)
#print(simplices)

def load_one_img():
    img = plt.imread(img_dir + "images" + "/" + "IMG_4026.JPG")
    plt.imshow(img, cmap = plt.cm.binary)
    plt.show()

def load_one_pose():
    pose = poses_bounds[0][ :-2].reshape(3, 5)
    c_to_w = pose[ : , :-1]
    hwf = pose[ : , -1]
    height, width, focal = hwf
    near, far = poses_bounds[0][-2: ]
    #print(hwf)
    print(pose)
    #print(height, width, focal)

#load_one_img()
#load_one_pose()
'''
root_dir = img_dir + "images_8/"
file_list = []
pic_num = 0
for file in os.listdir(root_dir):
    file_list.append(os.path.join(root_dir, file))
    pic_num += 1

print(file_list)
'''
class NeRFDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = []
        self.pic_num = 0

        for file in os.listdir(root_dir):
            self.file_list.append(os.path.join(root_dir, file))
            self.pic_num += 1

    def __len__(self):
        return self.pic_num

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)

        sample = {"image": img, "pose": poses_bounds[idx]}
        return sample

train_dataset = NeRFDataset(root_dir = img_dir + "images_8/", transform = None)
train_dataloader = DataLoader(dataset = train_dataset, batch_size = 4, shuffle = True, num_workers = 4)
#print(poses_bounds[0])

'''
plt.figure()
for (cnt, i) in enumerate(train_dataset):
    image = i["image"]
    label = i["label"]

    ax = plt.subplot(5, 4, cnt+1)
    ax.axis('off')
    ax.imshow(image)
    ax.set_title('label {}'.format(label))
    plt.pause(1)
'''
