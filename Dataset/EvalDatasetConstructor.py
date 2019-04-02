from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn.functional as functional
import torch.utils.data as data
import time
from utils import *
import torch
import math


class EvalDatasetConstructor(data.Dataset):
    def __init__(self,
                 data_dir_path,
                 gt_dir_path,
                 validate_num,
                 mode
                 ):
        self.validate_num = validate_num
        self.imgs = []
        self.data_root = data_dir_path
        self.gt_root = gt_dir_path
        self.calcu = HSI_Calculator()
        self.mode = mode
        self.GroundTruthProcess = GroundTruthProcess(1, 1, 2).cuda()
        for i in range(self.validate_num):
            img_name = '/IMG_' + str(i + 1) + ".jpg"
            gt_map_name = '/GT_IMG_' + str(i + 1) + ".npy"
            img = Image.open(self.data_root + img_name).convert("RGB")
            height = img.size[1]
            width = img.size[0]
            if height <= 400:
                resize_height = 512
            else:
                resize_height = math.ceil(height / 128) * 128
                
            if width <= 400:
                resize_width = 512
            else:
                resize_width = math.ceil(width / 128) * 128
            img = transforms.Resize([resize_height, resize_width])(img)
            gt_map = Image.fromarray(np.squeeze(np.load(self.gt_root + gt_map_name)))
            self.imgs.append([img, gt_map])

    def __getitem__(self, index):
        if self.mode == 'crop':
            img, gt_map = self.imgs[index]
            img = transforms.ToTensor()(img).cuda()
            gt_map = transforms.ToTensor()(gt_map).cuda()
            img_shape = img.shape  # C, H, W
            gt_shape = gt_map.shape
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            patch_height = (img_shape[1]) // 2
            patch_width = (img_shape[2]) // 2
            imgs = []
            for i in range(3):
                for j in range(3):
                    start_h = (patch_height // 2) * i
                    start_w = (patch_width // 2) * j
                    imgs.append(img[:, start_h:start_h + patch_height, start_w:start_w + patch_width])
            imgs = torch.stack(imgs)
            gt_map = self.GroundTruthProcess(gt_map.view(1, *(gt_shape)))
            return index + 1, imgs, gt_map.view(1, gt_shape[1] // 2, gt_shape[2] // 2)

    def __len__(self):
        return self.validate_num
