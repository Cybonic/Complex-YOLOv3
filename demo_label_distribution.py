

import os, sys, time, datetime, argparse
from utils.utils import *

import tqdm
import utils.config as cnf
from utils.kitti_yolo_dataset import KittiYOLODataset
import numpy as np 

def unique(list1): 
    # intilize a null list 
    unique_list = [] 
      
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    # print list 
    #for x in unique_list: 
        #print(x) 
    return(unique_list)
      
def label_distribution(split,folder):

    dataset = KittiYOLODataset(cnf.root_dir, split=split, mode='EVAL', folder = folder, data_aug=False)
    frames = dataset.image_idx_list.__len__()
    labels_distribution = dict()
    # go through all frames
    for i in tqdm.tqdm(range(0,frames)):
        # get targets
        _, imgs, targets = dataset.__getitem__(i)
        labels = targets[:, 1].tolist()
        uniq_labels = unique(labels)

        # get through all targets
        for single_label in uniq_labels:
            
            index = np.where(np.array(labels) == single_label)

            n_objects = len(index[0])
            # update label stats 
            if(single_label in labels_distribution.keys()):
                labels_distribution[single_label] += n_objects
            else:
                labels_distribution.update({single_label : n_objects})

    labels_coresp = cnf.CLASS_NAME_TO_ID
    label_value = np.array(list(labels_coresp.values()))
    label_names  = np.array(list(labels_coresp.keys()))

    total = 0
    # compute total targets
    for label in labels_distribution.items():
        total += label[1]
    # plot stats
    for label_dist in labels_distribution.items():
        idx = int(np.where(label_value == label_dist[0])[0])

        label_name      = label_names[idx]
        label_dist_perc = round(label_dist[1]/total,2)*100
        label_dist      = label_dist[1]

        print(label_name + " : " + str(label_dist_perc) + "% : " + str(label_dist))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_path", type=str, default="data/classes.names", help="path to class label file")
    opt = parser.parse_args()
    print(opt)


    class_names = load_classes(opt.class_path)

    split='valid_00'
    folder = '00'
    label_distribution(split,folder)