# Basic libraries
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class SimCLRNetwork(nn.Module):
    """
    Class SimCLRNetwork
    """
    def __init__(self, z_dim=128):
        """
        Constructor class
        """
        super(SimCLRNetwork, self).__init__()
        self.z_dim = z_dim
        # base encoder based on resnet50
        self.base_encoder = models.resnet50()
        nb_features = self.base_encoder.fc.in_features
        self.base_encoder.fc = nn.Identity()
        # projection head, a two layer MLP
        self.projection_head = nn.Sequential(
            nn.Linear(nb_features, nb_features),
            nn.ReLU(),
            nn.Linear(nb_features, self.z_dim),
        )

    def get_z_dim(self):
        """
        return the latent space dimension
        """
        return self.z_dim

    def get_base_encoder_model(self):
        """
        return the base encoder model
        """
        return self.base_encoder

    def get_projection_head_model(self):
        """
        return the projection head model
        """
        return self.projection_head

    def forward(self, x1, x2):
        """
        Forward function
        Inputs:
            x1, x2: positive pair of image
        Returns:
            z1, z2: space vector of image x1, x2
        """
        # first augmentation
        h1 = self.base_encoder(x1)
        z1 = self.projection_head(h1)
        z1_normalized = F.normalize(z1, dim = 1)
        # second augmentation
        h2 = self.base_encoder(x2)
        z2 = self.projection_head(h2)
        z2_normalized = F.normalize(z2, dim = 1)
        return z1_normalized, z2_normalized

    def get_latent_space(self, x1):
        """
        Forward function on one image
        Inputs:
            x1: image
        Returns:
            z1: space vector of image x1
        """
        h1 = self.base_encoder(x1)
        z1 = self.projection_head(h1)
        z1_normalized = F.normalize(z1, dim = 1)
        return z1_normalized