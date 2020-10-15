# Basic libraries
import glob
import torch
from torchvision import datasets, transforms

# Custom libraries
import data_augmentation
from simclr_framework import SimCLRFramework
import tools

testset = datasets.STL10(
    '../data/', download=False, split="test",
    transform=data_augmentation.t_compose_resize(tools.OUTPUT_SIZE)) 
testloader = torch.utils.data.DataLoader(
    testset, shuffle=True, batch_size=tools.BATCH_SIZE, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
framework = SimCLRFramework(testloader, tools.BATCH_SIZE, device)
framework.load_model(sorted(glob.glob(tools.MODEL_PATH + "*"))[0])
tools.plot_latent_space(testloader, framework.simclr_network, device)