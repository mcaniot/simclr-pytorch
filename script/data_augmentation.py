# Basic libraries
import cv2
import numpy as np
import torch
from torchvision import transforms

# Global variables
SIGMA_MIN = 0.1
SIGMA_MAX = 2.0
PROB_BLUR = 0.5

# Data augmentation functions

def t_compose_simclr(input_size):
    """
    Compose all transformation needed for data augmentation
    inputs:
        input_size: int, define the size of the input image.
    Return:
        a transform compose
    """
    return transforms.Compose([
        t_random_resized_crop(input_size),
        t_color_distortion(),
        t_gaussian_blur(),
        transforms.ToTensor()
    ])

def t_random_resized_crop(input_size):
    """
    Random crop and resize an image
    Input:
        input_size: int, size of the input image.
    Return:
        a transform compose
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=input_size,
            scale=(0.08, 1.0),
            ratio=(3.0/4.0, 4.0/3.0)),
        transforms.RandomHorizontalFlip(p=0.5)
    ])

def t_color_distortion(color_strength=1.0):
    """
    Color distorsion implmentation, cf https://arxiv.org/abs/2002.05709
    Input:
        color_strength: strength of the color distorsion
    Return:
        a transform compose
    """
    color_jitter = transforms.ColorJitter(
        0.8*color_strength,
        0.8*color_strength,
        0.8*color_strength,
        0.2*color_strength)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    return transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    
def gaussian_blur(img):
    """
    Applied a gaussian blur on a image with a probality fixed.
    Input:
        img: PIL Image
    Return:
        an image with the gaussian blur applied
    """
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
    """
    Transform of the function gaussian blur
    Return:
        a transform based on the function gaussian_blur
    """
    return transforms.Lambda(gaussian_blur)

# Class

class CreatePosPair():
    """
    Class for creating positive pair dataset
    """
    def __init__(self, input_size):
        self.compose_tranform = t_compose_simclr(input_size)

    def __call__(self, x):
        return self.compose_tranform(x), self.compose_tranform(x)