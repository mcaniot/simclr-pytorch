import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import data_augmentation
from simclr_framework import SimCLRFramework

BATCH_SIZE = 128

trainset = datasets.STL10(
    '../data/', download=False, split="unlabeled",
    transform=data_augmentation.CreatePosPair(96))
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
framework = SimCLRFramework(trainloader, BATCH_SIZE, device)
framework.train()
framework.save_model()