import cv2
from segmentation_model import DataLoader, Model
from utils import encode_segmap, decode_segmap, label_colors
from torchvision import transforms
from PIL import Image
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

car_styles = ["car1", "car2", "car3"]
road_styles = ["road1", "road2", "road3"]

convert_tensor = transforms.ToTensor()
transform = transforms.ToPILImage()

def swap_style(image_path, style, car):

    # main_image = cv2.imread('stylize-datasets/black.png')
    main_image = cv2.imread('stylize-datasets/images/' + image_path)
    main_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB)

    tmp_path = image_path.split('_')

    map = cv2.imread('stylize-datasets/maps/' + '_'.join(tmp_path[:-1]) + '_gtFine_color.png')
    map = cv2.cvtColor(map, cv2.COLOR_BGR2RGB)
    # cv2.imshow('image',map)
    # plt.imshow(map)
    # plt.show()

    if not os.path.exists('stylize-datasets/output/' + image_path[:-4] + '-stylized-' + style + '.png'):
        return False

    stylized_image = cv2.imread('stylize-datasets/output/' + image_path[:-4] + '-stylized-' + style + '.png')
    # print('stylize-datasets/output/' + image_path[:-4] + '-stylized-' + style + '.png')
    stylized_image = cv2.cvtColor(stylized_image, cv2.COLOR_BGR2RGB)

    map = np.mean(map, axis=2)
    # map = reverse_decoding(map)
    # return transform(decode_segmap(map))

    map = convert_tensor(map)
    main_image = convert_tensor(main_image)
    stylized_image = convert_tensor(stylized_image)

    if car:
        match = 142 / 3
    else:
        match = 320 / 3


    # print(map[100][100], map[100][100] == 70)

    mytensor = map == match

    # print(map)

    newImage = torch.where(mytensor, stylized_image, main_image)

    return transform(newImage)



if __name__ == '__main__':
    # root_dir = 'data'
    # mode = 'gtFine'
    # split = 'train'
    # label_path = os.path.join(os.getcwd(), root_dir+'/'+mode+'/'+split)
    # rgb_path = os.path.join(os.getcwd(), root_dir+'/leftImg8bit/'+split)
    # city_list = os.listdir(rgb_path)
    # for city in city_list:
    #     if city == '.DS_Store':
    #         continue
    #     print(city)
    # os.mkdir("train/"+city)
    temp = os.listdir('stylize-datasets/images')
    list_items = temp.copy()

    # 19-class label items being filtered
    for item in temp:
        if not item.endswith('leftImg8bit.png', 0, len(item)):
            continue
        else:
            for style in car_styles:
                image = swap_style(item,style, False)
                if image:
                    image.save("stylize-datasets/"+'stylized_images1/' + item[:-4] + '-road-' + style + '.png')
                    break;
            for style in road_styles:
                image = swap_style(item,style, True)
                if image:
                    image.save("stylize-datasets/"+'stylized_images1/' + item[:-4] + '-car-' + style + '.png')
                    break;
            # break;
        # break;

