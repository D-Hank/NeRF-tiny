import imghdr
from logging import root
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image
from matplotlib.image import imread
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class NeRFDataset(Dataset):
    def __init__(self, root_dir, low_res = 8, transform = None):
        self.root_dir = root_dir
        self.low_res = low_res
        self.transform = transform
        self.file_list = []
        self.pic_num = 0
        self.poses_bounds = np.load(root_dir + "poses_bounds.npy")

        if low_res == None:
            img_dir = root_dir + "images/"
        else:
            img_dir = root_dir + "images_" + str(low_res) + "/"

        for file in os.listdir(img_dir):
            self.file_list.append(os.path.join(img_dir, file))
            self.pic_num += 1

    def __len__(self):
        return self.pic_num

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = imread(img_path)
        poses_bound = self.poses_bounds[idx]

        #print(poses_bounds[idx])
        return img, poses_bound

'''
train_dataset = NeRFDataset(root_dir = img_dir + "images_8/", transform = None)
train_dataloader = DataLoader(dataset = train_dataset, batch_size = 4, shuffle = True, num_workers = 4)
#print(poses_bounds[0])

plt.figure()
for (cnt, i) in enumerate(train_dataset):
    image, label = i
    #ax = plt.subplot(5, 4, cnt+1)
    #ax.axis('off')
    #ax.imshow(image)
    #ax.set_title('label {}'.format(label))
    #plt.pause(1)
    #with open("img.txt","w") as f:
    #    f.write("\n".join(" ".join(map(str, x)) for x in image))
    print(image)
    exit(0)
'''

