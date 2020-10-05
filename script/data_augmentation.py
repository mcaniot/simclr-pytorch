#data augmentation
from torchvision import transforms
import torch
import numpy as np
import cv2


SIGMA_MIN = 0.1
SIGMA_MAX = 2.0
PROB_BLUR = 0.5

def t_compose_simclr(input_size):
    return transforms.Compose([
        t_random_resized_crop(input_size),
        t_color_distortion(),
        t_gaussian_blur(),
        transforms.ToTensor()
    ])

def t_random_resized_crop(input_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=input_size,
            scale=(0.08, 1.0),
            ratio=(3.0/4.0, 4.0/3.0)),
        transforms.RandomHorizontalFlip(p=0.5)
    ])

def t_color_distortion(s=1.0):
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    return transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    
def gaussian_blur(img):
    if np.random.uniform(0, 1) > PROB_BLUR:
        kernel = int(img.size[1] * 0.1)
        sigma = np.random.uniform(SIGMA_MIN, SIGMA_MAX)
        image = np.array(img)
        image_blur = cv2.GaussianBlur(
            image,
            (kernel, kernel),
            sigma
        )
        new_image = image_blur
        return new_image
    else:
        return img

def t_gaussian_blur():
    return transforms.Lambda(gaussian_blur)


class CreatePosPair():
    def __init__(self, input_size):
        self.compose_tranform = t_compose_simclr(input_size)

    def __call__(self, x):
        return self.compose_tranform(x), self.compose_tranform(x)