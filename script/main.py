import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import data_augmentation
import simclr

BATCH_SIZE = 64

trainset = datasets.STL10(
    '../data/', download=False, split="unlabeled",
    transform=data_augmentation.CreatePosPair(96))
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
framework = simclr.SimCLR(trainloader, BATCH_SIZE, device)
framework.train()
framework.save_model()
def test(dataiter):
    images, labels = dataiter.next()
    image = images[0]
    t_c_s = data_augmentation.t_compose_simclr(image.shape[1])
    image = transforms.ToPILImage()(image)
    image = t_c_s(image)
    print(image.shape)
    plt.imshow(image.permute(1,2,0))
    plt.show()