import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import lars
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import data_augmentation

NB_EPOCHS = 100
TEMPERATURE = 0.5

class SIMCLRNetwork(nn.Module):
    def __init__(self, z_dim=128):
        super(SIMCLRNetwork, self).__init__()
        self.base_encoder = models.resnet50(pretrained=False)
        nb_features = self.base_encoder.fc.out_features
        self.projection_head = nn.Sequential(
            nn.Linear(nb_features, nb_features),
            nn.ReLU(),
            nn.Linear(nb_features, z_dim),
        )
        
    def forward(self, x1, x2):
        h1 = self.base_encoder.forward(x1)
        z1 = self.projection_head.forward(h1)
        # second augmentation
        h2 = self.base_encoder.forward(x2)
        z2 = self.projection_head.forward(h2)
        return z1, z2


class SimCLR(object):
    def __init__(self, dataset, batch_size, device):
        self.device = device
        self.dataset = dataset
        self.batch_size = batch_size
        self.simclr_model = SIMCLRNetwork().to(device)
        self.writer = SummaryWriter("../logs/SIMCLR")
    def pairwise_similarity(self, z1, z2):
        z = torch.cat((z1, z2), dim=0).to(self.device)
        return torch.nn.CosineSimilarity(dim=2)(
            z.unsqueeze(1), z.unsqueeze(0)).to(self.device)
    def compute_loss(self, z1, z2):
        # pairwise similarity
        pairwise_sim = self.pairwise_similarity(z1, z2)
        pos_pair = torch.exp((pairwise_sim) / TEMPERATURE).to(self.device)
        return torch.mean(
            -torch.log((pos_pair)/torch.sum(pos_pair))).to(self.device)
    def train(self, log_interval=10):
        optimizer = lars.LARS(
            self.simclr_model.parameters(),
            lr=0.3*self.batch_size/256,
            weight_decay=1e-6)
        progress_bar = tqdm(
            total=NB_EPOCHS*len(self.dataset),
            desc="SIMCLR training",
            leave=False)
        total_loss, train_loss, total_num = 0.0, 0.0, 0.0
        for epoch in range(NB_EPOCHS):
            self.simclr_model.train()
            print("num epoch: " + str(epoch) +  "/" + str(NB_EPOCHS))
            for batch_id, ((x1, x2), _) in enumerate(self.dataset):
                # send to device
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)

                # forward
                z1, z2 = self.simclr_model.forward(x1, x2)

                # send to device
                z1 = z1.to(self.device)
                z2 = z2.to(self.device)

                # loss
                loss_pos_pair = self.compute_loss(z1, z2)
                optimizer.zero_grad()
                loss_pos_pair.backward()
                optimizer.step()
                total_num += self.batch_size
                loss = loss_pos_pair.item()
                train_loss += loss
                total_loss += loss * self.batch_size
                if batch_id % log_interval == 0:
                    self.writer.add_scalar(
                        'loss/training_loss_generator',
                        train_loss / log_interval,
                        epoch * len(self.dataset) + batch_id)
                    train_loss = 0
                # progress_bar.set_description(
                #     'Train Epoch: [{}/{}] Loss: {:.4f}'.format(
                #         epoch, NB_EPOCHS, total_loss / total_num))
                # progress_bar.set_description(
                #     'Train Batch: [{}/{}] Loss: {:.4f}'.format(
                #         batch_id, len(self.dataset), total_loss / total_num))
                progress_bar.update(1)
        return total_loss / total_num
    def save_model(self):
        torch.save(
            self.simclr_model.state_dict(),
            "../models/simclr.pth")

            
