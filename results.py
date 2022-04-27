from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms, datasets
import torch
from torch.utils.data import DataLoader, Dataset
from segmentation_model import DataLoader, Model
from utils import encode_segmap, decode_segmap
import numpy as np

event_acc1 = EventAccumulator('lightning_logs/version_3/events.out.tfevents.1647370160.dhcp-10-250-32-83.harvard.edu.1603.0')
event_acc12 = EventAccumulator('lightning_logs/version_10/events.out.tfevents.1649357397.dhcp-10-250-196-68.harvard.edu.52945.0')
event_acc2 = EventAccumulator('lightning_logs/version_4/events.out.tfevents.1647437317.dhcp-10-250-32-83.harvard.edu.4635.0')
event_acc22 = EventAccumulator('lightning_logs/version_11/events.out.tfevents.1649430596.dhcp-10-250-196-68.harvard.edu.55814.0')
event_acc3 = EventAccumulator('lightning_logs/version_5/events.out.tfevents.1647494430.dhcp-10-250-32-83.harvard.edu.14436.0')
event_acc32 = EventAccumulator('lightning_logs/version_12/events.out.tfevents.1649486644.dhcp-10-250-196-68.harvard.edu.57889.0')

event_acc1.Reload()
event_acc12.Reload()
event_acc2.Reload()
event_acc22.Reload()
event_acc3.Reload()
event_acc32.Reload()


def show_iou():

    # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
    w_times, step_nums, base_vals = zip(*event_acc1.Scalars('val_iou'))
    w_times, step_nums, base_vals1 = zip(*event_acc12.Scalars('val_iou'))
    w_times, step_nums, blur_vals = zip(*event_acc2.Scalars('val_iou'))
    w_times, step_nums, blur_vals1 = zip(*event_acc22.Scalars('val_iou'))
    w_times, step_nums, sharp_vals = zip(*event_acc3.Scalars('val_iou'))
    w_times, step_nums, sharp_vals1 = zip(*event_acc32.Scalars('val_iou'))

    x = range(1,21)
    # print(base_vals, blur_vals, sharp_vals)
    plt.plot(x, base_vals + base_vals1)
    print(base_vals + base_vals1)
    plt.plot(x, blur_vals + blur_vals1)
    plt.plot(x, sharp_vals + sharp_vals1)
    # plt.plot(x, base_vals1)
    plt.legend(['Standard Cityscapes', 'Blurred Cityscapes', 'Sharpened Cityscapes', 'Next 10'])
    plt.title('Intersection over union scores by epoch on the validation set')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.show()
    # print(vals)

def show_loss():

    w_times, step_nums, base_vals = zip(*event_acc1.Scalars('val_loss'))
    w_times, step_nums, base_vals1 = zip(*event_acc12.Scalars('val_loss'))
    w_times, step_nums, blur_vals = zip(*event_acc2.Scalars('val_loss'))
    w_times, step_nums, blur_vals1 = zip(*event_acc22.Scalars('val_loss'))
    w_times, step_nums, sharp_vals = zip(*event_acc3.Scalars('val_loss'))
    w_times, step_nums, sharp_vals1 = zip(*event_acc32.Scalars('val_loss'))

    x = range(1,21)
    # print(base_vals, blur_vals, sharp_vals)
    plt.plot(x, base_vals + base_vals1)
    print(base_vals + base_vals1)
    plt.plot(x, blur_vals + blur_vals1)
    plt.plot(x, sharp_vals + sharp_vals1)
    # plt.plot(x, blur_vals)
    # plt.plot(x, sharp_vals)
    plt.legend(['Standard Cityscapes', 'Blurred Cityscapes', 'Sharpened Cityscapes'])
    plt.title('Dice loss scores by epoch on the validation set')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Loss')
    plt.show()

def show_predictions():
    transform = A.Compose(
    [
    A.Resize(256, 512),
    # A.HorizontalFlip(),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
    ]
    )
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])


    test_class = DataLoader('data/evaluation', split="val", mode='fine', target_type='semantic', transforms=transform)
    model = Model.load_from_checkpoint('models20/sharp.ckpt')

    test_loader=DataLoader(test_class, batch_size=1, shuffle=False)

    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            img,seg = batch
            output=model(img)
            break
    # print(img.shape, seg.shape, output.shape)

    sample = 0
    outputx=output.detach().cpu()[sample]
    encoded_mask = encode_segmap(seg[sample].clone())
    decoded_mask = decode_segmap(encoded_mask.clone())
    decoded_output = decode_segmap(torch.argmax(outputx,0))
    fig, ax = plt.subplots(ncols=3, figsize=(16,4))
    img = invTrans(img[sample])
    ax[0].imshow(np.moveaxis(img.numpy(),0,2))
    ax[1].imshow(decoded_mask)
    ax[2].imshow(decoded_output)
    ax[0].title.set_text('Image')
    ax[1].title.set_text('True Segmentation')
    ax[2].title.set_text('Predicted Segmentation')
    fig.suptitle('Predictions on Stylized Image from Sharpened Dataset Model')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    plt.show()

def show_example_images():
    import matplotlib.gridspec as gridspec
    import cv2


    fig, ax = plt.subplots(nrows=3, figsize=(18,6))
    fig.tight_layout()
    image = cv2.imread("base_data/leftImg8bit/train/aachen/aachen_000015_000019_leftImg8bit.png")
    ax[0].imshow(image)
    image = cv2.imread("blur_data/leftImg8bit/train/aachen/aachen_000015_000019_leftImg8bit.png")
    ax[1].imshow(image)
    image = cv2.imread("data/leftImg8bit/train/aachen/aachen_000015_000019_leftImg8bit.png")
    ax[2].imshow(image)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')


    plt.show()

def calculate_iou(data, model):
    import numpy as np
    from sklearn.metrics import jaccard_score
    transform = A.Compose(
    [
    A.Resize(256, 512),
    # A.HorizontalFlip(),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
    ]
    )
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])



    test_class = DataLoader('data/' + data, split="val", mode='fine', target_type='semantic', transforms=transform)
    model = Model.load_from_checkpoint('models20/' + model + '.ckpt')

    test_loader=DataLoader(test_class, batch_size=125, shuffle=False)

    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            img,seg = batch
            output=model(img)
            break
    # print(img.shape, seg.shape, output.shape)
    ious = 0
    for i in range(125):
            sample = i
            outputx=output.detach().cpu()[sample]
            encoded_mask = encode_segmap(seg[sample].clone())
            decoded_mask = decode_segmap(encoded_mask.clone()) * 100
            decoded_output = decode_segmap(torch.argmax(outputx,0)) * 100
            decoded_mask = decoded_mask.flatten().astype(int)
            decoded_output = decoded_output.flatten().astype(int)
            # print(decoded_mask[4000:4050])
            # print(decoded_output[4000:4050])
            iou = jaccard_score(decoded_mask, decoded_output, average='micro')
            ious = ious + iou
    return ious/125





# copy_stylized_data()
# create_maps()
# create_grayscale_data()
iou = calculate_iou('evaluation_gray', 'standard')
print(iou)

iou = calculate_iou('evaluation_gray', 'sharp')
print(iou)# create_maxps()

iou = calculate_iou('evaluation_gray', 'blur')
print(iou)
# # show_predictions()