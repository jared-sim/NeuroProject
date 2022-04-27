from torchvision.datasets import Cityscapes
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms, datasets
from utils import encode_segmap, decode_segmap


transform = A.Compose(
[
A.Resize(256, 512),
A.HorizontalFlip(),
A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
ToTensorV2()
]
)

from typing import Any, Callable, Dict, List, Optional, Union, Tuple

class DataLoader(Cityscapes):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert("RGB")

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            transformed = transform(image=np.array(image), mask=np.array(target))
        return transformed['image'], transformed['mask']



def view_image(img, seg):
    res=encode_segmap(seg.clone())
    print(res.shape)
    print(torch.unique(res))
    print(len(torch.unique(res)))
    res1=decode_segmap(res.clone())
    plt.imshow(res1)
    plt.show()


import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import segmentation_models_pytorch as smp
import multiprocessing
import torchmetrics
import torch

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.layer = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes
        )

        self.lr= 1e-4
        self.batch_size=10
        self.num_worker=6
        # self.train_ious = []
        # self.train_losses = []
        self.val_ious = []
        self.val_losses = []
        self.ious = []
        self.losses = []

        self.criterion= smp.losses.DiceLoss(mode='multiclass')
        self.metrics = torchmetrics.IoU(num_classes=n_classes)

        self.train_class = DataLoader("data/sharp_data/", split='train', mode='fine', target_type="semantic", transforms=transform)

        self.val_class = DataLoader("data/sharp_data/", split='val', mode='fine', target_type="semantic", transforms=transform)

    def process(self, image, segment):
        out=self(image)
        segment=encode_segmap(segment)
        loss=self.criterion(out,segment.long())
        iou=self.metrics(out,segment)
        return loss, iou


    def forward(self, x):
        return self.layer(x)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return opt

    def train_dataloader(self):
        return DataLoader(self.train_class, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker, pin_memory=True)

    def training_step(self, batch, batch_idx):
        image, segment = batch
        loss, iou = self.process(image, segment)
        # self.ious.append(iou.item())
        # self.losses.append(loss.item())
        # print('training', self.losses, self.ious)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_iou', iou, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def val_dataloader(self):
        return DataLoader(self.val_class, batch_size=self.batch_size, shuffle=False, num_workers=self.num_worker, pin_memory=True)

    def validation_step(self, batch, batch_idx):
        image, segment = batch
        loss, iou = self.process(image, segment)
        self.ious.append(iou.item())
        self.losses.append(loss.item())
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_epoch_end(self, batch):
    # outs is a list of whatever you returned in `validation_step`

        loss = sum(self.losses)/len(self.losses)
        iou = sum(self.ious)/len(self.ious)
        self.losses = []
        self.ious = []
        self.val_losses.append(loss)
        self.val_ious.append(iou)

if __name__ == '__main__':
    # dataset = DataLoader("data/blur_data", split='train', mode='fine', target_type="semantic", transforms=transform)

    model = Model()

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='basic_checkpoints1', filename='file', save_last=True)

    trainer = pl.Trainer(resume_from_checkpoint='./basic_checkpoints/sharp.ckpt', max_epochs=20, auto_lr_find=False, auto_scale_batch_size=False, callbacks=[checkpoint_callback])

    trainer.fit(model)

