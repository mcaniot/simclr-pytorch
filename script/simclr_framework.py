# Basic libraries
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision.models as models

# Custom libraries
import data_augmentation
import lars
from simclr_network import SimCLRNetwork

NB_EPOCHS = 100
TEMPERATURE = 0.5
MODEL_PATH = "../models/simclr_%s.pth"

class SimCLRFramework(object):
    """
    Class SimCLRFramework
    Framework introduced by https://arxiv.org/abs/2002.05709
    """
    def __init__(self, dataset, batch_size, device):
        """
        Constructor
        """
        self.device = device
        self.dataset = dataset
        self.batch_size = batch_size
        self.simclr_network = SimCLRNetwork().to(device)
        self.writer = SummaryWriter(
            "../logs/SIMCLR_%s" %((str(int(time.time())))))

    def pairwise_similarity(self, z1, z2):
        """
        Compute the pairwise similarity between z1 and z2
        Inputs:
            z1, z2: space vectors
        """
        z = torch.cat((z1, z2), dim=0).to(self.device)
        return torch.nn.CosineSimilarity(dim=2)(
            z.unsqueeze(1), z.unsqueeze(0)).to(self.device)

    def compute_loss(self, z1, z2):
        """
        Compute the loss using the pairwise similarity
        Inputs:
            z1, z2: space vectors
        Return
            the computed loss function
        """
        pairwise_sim = self.pairwise_similarity(z1, z2)
        pos_pair = torch.exp((pairwise_sim) / TEMPERATURE).to(self.device)
        return torch.mean(
            -torch.log((pos_pair)/torch.sum(pos_pair))).to(self.device)

    def train(self, log_interval=10):
        """
        train the model using SimCLR framework
        Inputs:
            log_interval: number of batch between logs
        """
        optimizer = lars.LARS(
            self.simclr_network.parameters(),
            lr=0.3*self.batch_size/256,
            weight_decay=1e-6)
        progress_bar = tqdm(
            total=NB_EPOCHS*len(self.dataset),
            desc="SIMCLR training",
            leave=False)
        total_loss, train_loss, total_num = 0.0, 0.0, 0.0
        for epoch in range(NB_EPOCHS):
            self.simclr_network.train()
            print("num epoch: " + str(epoch) +  "/" + str(NB_EPOCHS))
            for batch_id, ((x1, x2), _) in enumerate(self.dataset):
                # send to device
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)

                # forward
                z1, z2 = self.simclr_network.forward(x1, x2)

                # send to device
                z1 = z1.to(self.device)
                z2 = z2.to(self.device)

                # loss
                loss_pos_pair = self.compute_loss(z1, z2)

                optimizer.zero_grad()
                loss_pos_pair.backward()
                optimizer.step()

                # Logs
                loss = loss_pos_pair.item()
                train_loss += loss
                if batch_id % log_interval == 0:
                    self.writer.add_scalar(
                        'loss/training_loss_generator',
                        train_loss / log_interval,
                        epoch * len(self.dataset) + batch_id)
                    train_loss = 0
                progress_bar.update(1)
        return self.simclr_network.get_base_encoder_model()

    def save_model(self):
        """
        save the train model
        """
        torch.save(
            self.simclr_network.state_dict(),
            MODEL_PATH %(str(int(time.time()))))
