import torch
from PIL import Image
import os
import pdb
import numpy as np
import cv2
from data.mytransforms import find_start_pos
from lib.utils import letterbox_for_img

from tqdm import tqdm
import json

def loader_func(path):
    return Image.open(path)


class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None):
        super(LaneTestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        with open(list_path, 'r') as f:
            self.list = f.readlines()
        self.list = [l[1:] if l[0] == '/' else l for l in self.list]  # exclude the incorrect path prefix '/' of CULane
        self.img_size = 640

    def __getitem__(self, index):
        name = self.list[index].split()[0]
        img_path = os.path.join(self.path, name)
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)  # BGR
        #img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        assert img is not None, 'Image Not Found ' + img_path

        h0, w0 = img.shape[:2]
     

        # Padded resize
        re_img, ratio, pad = letterbox_for_img(img, new_shape=self.img_size, auto=True)
        h, w = re_img.shape[:2]
        shapes = (h0, w0), ((h / h0, w / w0), pad)

        # Convert
        #img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        re_img = np.ascontiguousarray(re_img)


        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        # return path, img, img0, self.cap, shapes

        if self.img_transform is not None:
            img = self.img_transform(img)
            re_img = self.img_transform(re_img)
            
        # if re_img.ndimension() == 3:
        #     re_img = re_img.unsqueeze(0)

        return re_img, name, img, shapes

    def __len__(self):
        return len(self.list)


