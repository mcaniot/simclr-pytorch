# Basic libraries
import torch
from torchvision import datasets

# Custom libraries
import data_augmentation
from simclr_framework import SimCLRFramework
import tools

trainset = datasets.STL10(
    '../data/', download=False, split="unlabeled",
    transform=data_augmentation.CreatePosPair(tools.OUTPUT_SIZE))
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=tools.BATCH_SIZE, shuffle=True, drop_last=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
framework = SimCLRFramework(trainloader, tools.BATCH_SIZE, device)
framework.train()

