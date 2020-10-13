# Basic libraries
import time
from tqdm import tqdm
import torch
from torch import optim
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
        # self.lr = 0.3*self.batch_size/256
        self.lr = 1e-3
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.weight_decay = 1e-6
        self.writer = SummaryWriter(
            "../logs/SIMCLR_%s" %((str(int(time.time())))))
        self.temperature = TEMPERATURE

    def compute_loss_NT_Xent(self, z_i, z_j):
        batch_size = z_i.shape[0]
        assert batch_size == z_j.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        # cosine similarity on all different pairs in z
        sim_matrix = torch.exp(nn.CosineSimilarity(dim=2)(
            z.unsqueeze(1), z.unsqueeze(0)) / self.temperature)

        # remove all cosine similarity between same augmented images
        mask = (torch.ones_like(sim_matrix) - torch.eye(
            2 * batch_size, device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(
            2 * batch_size, -1)

        # cosine similarity on positive pair z_i and z_j
        pos_sim = torch.exp(nn.CosineSimilarity(dim=1)(
            z_i, z_j) / self.temperature)

        # loss_i_j = loss_j_i concatenate cosine similarity for a positive pair
        # z_i and z_j
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

        # Final loss
        loss = torch.mean(-torch.log(pos_sim / torch.sum(sim_matrix,dim=-1)))    
        return loss

    def train(self, log_interval=10):
        """
        train the model using SimCLR framework
        Inputs:
            log_interval: number of batch between logs
        """
        # optimizer = lars.LARS(
        #     self.simclr_network.parameters(),
        #     lr=self.lr,
        #     weight_decay=self.weight_decay)
        optimizer = optim.Adam(
            self.simclr_network.parameters(),
            lr=self.lr)
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
                batch_loss = self.compute_loss_NT_Xent(z1, z2)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # Logs
                loss = batch_loss.item()
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
