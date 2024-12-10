import os.path

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from os import listdir, walk
from os.path import join
from random import randint
import random
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, Resize, RandomHorizontalFlip


def random_horizontal_flip(imgs):
    if random.random() < 0.3:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
    return imgs

def random_rotate(imgs):
    if random.random() < 0.3:
        max_angle = 10
        angle = random.random() * 2 * max_angle - max_angle
        # print(angle)
        for i in range(len(imgs)):
            img = np.array(imgs[i])
            w, h = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
            imgs[i] = Image.fromarray(img_rotation)
    return imgs

def random_hsv(imgs):
    if random.random() < 0.3:
        hue = random.random() * 1. - .5  # (.5, .5)
        sat = random.random() * .4 + .8  # (0.8, 1.2)
        val = random.random() * .4 + .8  # (0.8, 1.2)
        for index, img in enumerate(imgs):
            img = TF.pil_to_tensor(img)
            img = TF.adjust_hue(img, hue)
            img = TF.adjust_saturation(img, sat)
            img = TF.adjust_brightness(img, val)
            imgs[index] = TF.to_pil_image(img)
    return imgs

def is_image(filename):
    return any(filename.endswith(extention) for extention in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP'])

def image_resize(load_size, training=True):
    if not training:
        load_size = (1536, 2048)
    return Compose([
        Resize(size=load_size, interpolation=Image.BICUBIC),
        ToTensor(),
    ])

class ErasingData(Dataset):
    def __init__(self, dataRoot, loadSize, training=True, better_data_aug=True):
        super(ErasingData, self).__init__()
        self.imageFiles = [join (dataRootK, files) for dataRootK, dn, filenames in walk(dataRoot) \
                           for files in filenames if is_image(files)]
        self.loadSize = loadSize
        self.img_transform = image_resize(loadSize, training)
        self.training = training
        self.better_data_aug = better_data_aug
    
    def __getitem__(self, index):
        img = Image.open(self.imageFiles[index])
        mask = Image.open(self.imageFiles[index].replace('all_images','mask'))
        stroke_mask = Image.open(self.imageFiles[index].replace('all_images','stroke_mask').replace('jpg', 'png'))
        gt = Image.open(self.imageFiles[index].replace('all_images','all_labels'))

        if self.training:
            all_input = [img, mask, stroke_mask, gt]
            all_input = random_horizontal_flip(all_input)
            all_input = random_rotate(all_input)
            if self.better_data_aug: all_input = random_hsv(all_input)
            img = all_input[0]
            mask = all_input[1]
            stroke_mask = all_input[2]
            gt = all_input[3]
        ### for data augmentation
        input_image = self.img_transform(img.convert('RGB'))
        mask = self.img_transform(mask.convert('RGB'))
        stroke_mask = self.img_transform(stroke_mask.convert('RGB'))
        ground_truth = self.img_transform(gt.convert('RGB'))
        path = self.imageFiles[index].split('/')[-1]

        return input_image, ground_truth, mask, stroke_mask, path
    
    def __len__(self):
        return len(self.imageFiles)

class EvalData(Dataset):
    def __init__(self, data_root, gt_root, load_size=512):
        super(EvalData, self).__init__()
        self.image_files = [join (dataRootK, files) for dataRootK, dn, filenames in walk(data_root) \
                            for files in filenames if is_image(files)]
        self.gt_root = gt_root
        self.load_size = load_size
        self.img_transform = image_resize(load_size, training=False)
    
    def __getitem__(self, index):
        img = Image.open(self.image_files[index])
        gt = Image.open(os.path.join(self.gt_root, os.path.basename(self.image_files[index])).replace('png', 'jpg'))
        inputImage = self.img_transform(img.convert('RGB'))

        groundTruth = self.img_transform(gt.convert('RGB'))
        path = self.image_files[index].split('/')[-1]

        return inputImage, groundTruth, path
    
    def __len__(self):
        return len(self.image_files)
