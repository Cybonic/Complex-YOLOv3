from __future__ import division

import numpy as np
import cv2
import torch

import utils.kitti_bev_utils as bev_utils
from utils.kitti_yolo_dataset import KittiYOLODataset
from torch.utils.data import DataLoader
import utils.config as cnf

if __name__ == "__main__":

    img_size=cnf.BEV_WIDTH

    # Get dataloader
    dataset = KittiYOLODataset(
        cnf.root_dir,
        split='train_00',
        mode='TRAIN',
        folder='00',
        data_aug=True,
    )

    # Load Dataset
    dataloader = DataLoader(
        dataset,
        1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    for batch_i in range(0,dataset.sample_id_list.__len__()):

        img_file, img, targets = dataset.__getitem__(batch_i)
        if(targets.shape[1]<2):
            print("no target found")


