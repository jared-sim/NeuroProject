import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import torch
import os

convert_tensor = transforms.ToTensor()
transform = transforms.ToPILImage()


def generate_blur(image):
    kernel = np.ones((8,8),np.float32)/64
    dst = cv2.filter2D(image,-1,kernel)
    return dst


def generate_blurred_edges(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    segMap = generate_segmentation_map(image_path)
    blurrMap = generate_blur(segMap)
    blurrImg = generate_blur(image)

    img = convert_tensor(image)
    sMap = convert_tensor(segMap)
    bMap = convert_tensor(blurrMap)
    bImg = convert_tensor(blurrImg)

    mytensor = (sMap - bMap) == 0

    newImage = torch.where(mytensor, img, bImg)

    return transform(newImage), transform(sMap)

if __name__ == '__main__':
    root_dir = 'data'
    mode = 'gtFine'
    split = 'train'
    label_path = os.path.join(os.getcwd(), root_dir+'/'+mode+'/'+split)
    rgb_path = os.path.join(os.getcwd(), root_dir+'/leftImg8bit/'+split)
    city_list = os.listdir(rgb_path)
    for city in city_list:
        if city == '.DS_Store':
            continue
        print(city)
        os.mkdir("train/"+city)
        temp = os.listdir(rgb_path+'/'+city)
        list_items = temp.copy()

        # 19-class label items being filtered
        for item in temp:
            if not item.endswith('leftImg8bit.png', 0, len(item)):
                continue
            else:
                image_path = rgb_path + '/' + city + '/' + item
                tmp_item = item.split('_')
                tmp_item = tmp_item[:-1] + ['gtFine', 'color.png']
                tmp_item = '_'.join(tmp_item)
                seg_path = label_path + '/' + city + '/' + tmp_item
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                segmentation = cv2.imread(seg_path)
                segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2RGB)
                blurrMap = generate_blur(segmentation)
                blurrImg = generate_blur(image)
                img = convert_tensor(image)
                sMap = convert_tensor(segmentation)
                bMap = convert_tensor(blurrMap)
                bImg = convert_tensor(blurrImg)


                mytensor = (sMap - bMap) == 0

                newImage = transform(torch.where(mytensor, img, bImg))
                img = transform(img)
                sMap = transform(sMap)
                newImage.save("train/"+city + '/' + item)