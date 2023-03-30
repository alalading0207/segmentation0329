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
        target = torch.max(self.read_tif(os.path.join(self.root, "Labelimgs", self.item_list[index].strip())) > 0, 0, True)[0].float()
        image, target = torch.split(self.transforms(torch.concat([image, target], 0)), [3, 1], 0)
        downsample_boundary_2 = self.get_downsample_boundary(target, [2,2], [2,2], 0.5)
        downsample_boundary_4 = self.get_downsample_boundary(target, [4,4], [4,4], 0.25)
        downsample_boundary_8 = self.get_downsample_boundary(target, [8,8], [8,8], 0.125)
        return image, target, downsample_boundary_2, downsample_boundary_4, downsample_boundary_8

    def __len__(self):
        return len(self.item_list)
    
    def read_tif(self, path):
        return torch.Tensor(cv2.imread(path, 1).transpose((2,0,1)).astype(np.float32))
    
    def get_downsample_boundary(self, input, kernel_size, stride, lower_bound, upper_bound=1):
        boundary = torch.nn.functional.avg_pool2d(input, kernel_size, stride)
        boundary = (boundary>lower_bound) & (boundary < upper_bound)
        return boundary.float()