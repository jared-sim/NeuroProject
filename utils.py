from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms, datasets
import torch
from torch.utils.data import DataLoader, Dataset
from segmentation_model import DataLoader, Model
import numpy as np

ignore_index=255
void_classes = [0,1,2,3,4,5,6,9,10,14,15,16,18,29,30,-1]
valid_classes = [ignore_index, 7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light',
                'traffic_sign', 'vegetation', 'terrain', 'sky', ' person', 'rider', 'car', 'truck',
                'bus', 'train', 'motorcycle', 'bicycle']

class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes=len(valid_classes)

colors = [[0,0,0],
            [128,64,128],
            [244,35,232],
            [70, 70, 70],
            [102,102,156],
            [190,153,153],
            [153,153,153],
            [250, 170,30],
            [220,220,0],
            [107,142,35],
            [152,251,152],
            [0,130,180],
            [220,20,60],
            [255, 0,0],
            [0,0,142],
            [0,0,70],
            [0,60,100],
            [0,80,100],
            [0,0,230],
            [119,11,32]]

label_colors = dict(zip(range(n_classes), colors))

def encode_segmap(mask):
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_index
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask

def decode_segmap(temp):
    temp = temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colors[l][0]
        g[temp == l] = label_colors[l][1]
        b[temp == l] = label_colors[l][2]
    rgb = np.zeros((temp.shape[0], temp.shape[1],3))
    # print(r)
    rgb[:,:,0] = r / 255.0
    rgb[:,:,1] = g / 255.0
    rgb[:,:,2] = b / 255.0
    return rgb


def create_maps():
    import shutil
    import os

    temp = os.listdir('stylize-datasets/images')
    list_items = temp.copy()

    # 19-class label items being filtered
    for item in temp:
        if not item.endswith('leftImg8bit.png', 0, len(item)):
            continue
        else:
            item = item[:-16]
            shutil.copy('data/base_data/gtFine/val/frankfurt/' + item + "_gtFine_color.png", 'data/evaluation_gray/gtFine/val/frankfurt/' + item + "_gtFine_color.png")
            shutil.copy('data/base_data/gtFine/val/frankfurt/' + item + "_gtFine_instanceIds.png", 'data/evaluation_gray/gtFine/val/frankfurt/' + item + "_gtFine_instanceIds.png")
            shutil.copy('data/base_data/gtFine/val/frankfurt/' + item + "_gtFine_labelIds.png", 'data/evaluation_gray/gtFine/val/frankfurt/' + item + "_gtFine_labelIds.png")
            shutil.copy('data/base_data/gtFine/val/frankfurt/' + item + "_gtFine_labelTrainIds.png", 'data/evaluation_gray/gtFine/val/frankfurt/' + item + "_gtFine_labelTrainIds.png")
            shutil.copy('data/base_data/gtFine/val/frankfurt/' + item + "_gtFine_polygons.json", 'data/evaluation_gray/gtFine/val/frankfurt/' + item + "_gtFine_polygons.json")

def copy_stylized_data():
    import shutil
    import os

    temp = os.listdir('stylize-datasets/output')
    list_items = temp.copy()
    print(temp)
    # 19-class label items being filtered
    for item in temp:
        shutil.copy('stylize-datasets/output/' + item, 'data/evaluation_stylized/leftImg8bit/val/frankfurt/' + item[:-15] + '.png')

def create_grayscale_data():
    from PIL import Image
    import os

    temp = os.listdir('stylize-datasets/images')
    list_items = temp.copy()

    # 19-class label items being filtered
    for item in temp:
        if not item.endswith('leftImg8bit.png', 0, len(item)):
            continue
        else:
            img = Image.open('stylize-datasets/images/' + item).convert('L')
            img.save('data/evaluation_gray/leftImg8bit/val/frankfurt/' + item)