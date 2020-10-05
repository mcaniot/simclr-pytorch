# Basic libraries
import torch.nn as nn
import torchvision.models as models


class SimCLRNetwork(nn.Module):
    """
    Class SimCLRNetwork
    """
    def __init__(self, z_dim=128):
        """
        Constructor class
        """
        super(SimCLRNetwork, self).__init__()
        # base encoder based on resnet50
        self.base_encoder = models.resnet50(pretrained=False)
        nb_features = self.base_encoder.fc.out_features
        # projection head, a two layer MLP
        self.projection_head = nn.Sequential(
            nn.Linear(nb_features, nb_features),
            nn.ReLU(),
            nn.Linear(nb_features, z_dim),
        )

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
        h1 = self.base_encoder.forward(x1)
        z1 = self.projection_head.forward(h1)
        # second augmentation
        h2 = self.base_encoder.forward(x2)
        z2 = self.projection_head.forward(h2)
        return z1, z2