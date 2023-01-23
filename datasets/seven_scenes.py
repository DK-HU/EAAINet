from __future__ import division
import os
import random
import numpy as np
import cv2
from torch.utils import data
import sys
import matplotlib.pyplot as plt

sys.path.append("..")
from datasets.utils import *


class SevenScenes(data.Dataset):
    def __init__(self, root, dataset='7S', scene='heads', split='train',
                 model='fdanet', aug='False'):
        self.intrinsics_color = np.array([[525.0, 0.0, 320.0],
                                          [0.0, 525.0, 240.0],
                                          [0.0, 0.0, 1.0]])

        self.intrinsics_depth = np.array([[585.0, 0.0, 320.0],
                                          [0.0, 585.0, 240.0],
                                          [0.0, 0.0, 1.0]])

        self.intrinsics_depth_inv = np.linalg.inv(self.intrinsics_depth)
        self.intrinsics_color_inv = np.linalg.inv(self.intrinsics_color)
        self.model = model
        self.dataset = dataset
        self.aug = aug
        self.root = os.path.join(root, '7Scenes')
        self.calibration_extrinsics = np.loadtxt(os.path.join(self.root,
                                                              'sensorTrans.txt'))
        self.scene = scene
        self.split = split
        self.obj_suffixes = ['.color.png', '.pose.txt', '.depth.png',
                             '.label.png']
        self.obj_keys = ['color', 'pose', 'depth', 'label']
        with open(os.path.join(self.root, '{}{}'.format(self.split,  # ./data/7Scenes/tarin或test.txt
                                                        '.txt')), 'r') as f:
            self.frames = f.readlines()
            if self.dataset == '7S' or self.split == 'test':
                self.frames = [frame for frame in self.frames \
                               if self.scene in frame]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index].rstrip('\n')
        scene, seq_id, frame_id = frame.split(' ')  # chess seq-03 frame-000000
        objs = {}
        objs['color'] = '/mnt/share/sda-2T/xietao/' + self.scene + '/' + seq_id + '/' + frame_id + '.color.png'
        objs['pose'] = '/mnt/share/sda-2T/xietao/' + self.scene + '/' + seq_id + '/' + frame_id + '.pose.txt'  # Twc
        objs['depth'] = '/mnt/share/sda-2T/xietao/' + self.scene + '/' + seq_id + '/' + frame_id + '.depth.png'
        img = cv2.imread(objs['color'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pose = np.loadtxt(objs['pose']) 
        if self.split == 'test':
            img, pose = to_tensor_query(img, pose)
            return img, pose 

        depth = cv2.imread(objs['depth'], -1)

        pose[0:3, 3] = pose[0:3, 3] * 1000

        depth[depth == 65535] = 0
        depth = depth * 1.0
        depth = get_depth(depth, self.calibration_extrinsics, self.intrinsics_color, self.intrinsics_depth_inv)

        coord, mask = get_coord(depth, pose, self.intrinsics_color_inv)
        img, coord, mask = data_aug(img, coord, mask, self.aug)

        coord = coord[4::8, 4::8, :]  # [60, 80, 3]
        mask = mask[4::8, 4::8].astype(np.float16)  # [60 80]
        img, coord, mask = to_tensor(img, coord, mask)

        return img, coord, mask


if __name__ == '__main__':
    datat = SevenScenes(root='../data/', split='train')
    trainloader = data.DataLoader(datat, batch_size=1, num_workers=1, shuffle=True, drop_last=True)  #

    for _, (img, coord, mask) in enumerate(trainloader):
        print(img.shape)
        print(coord.shape)
        print(mask.shape)
