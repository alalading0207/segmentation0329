import os
import numpy as np
import cv2
import torch

class DARBuDDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.item_list = list(open(os.path.join(root, "list.txt"), 'r').readlines())
        self.transforms = transforms
    def __getitem__(self, index):
        image = self.read_tif(os.path.join(self.root, "images", self.item_list[index].strip()))
        target = torch.max(self.read_tif(os.path.join(self.root, "Labelimgs", self.item_list[index].strip())) > 0, 0, True)[0]
        return torch.split(self.transforms(torch.concat([image, target], 0)), [3, 1], 0)


    def __len__(self):
        return len(self.item_list)
    
    def read_tif(self, path):
        return torch.Tensor(cv2.imread(path, 1).transpose((2,0,1)).astype(np.float32))