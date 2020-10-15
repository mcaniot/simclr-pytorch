# Basic libraries
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE

BATCH_SIZE = 128
OUTPUT_SIZE = 96
NB_EPOCHS = 100
TEMPERATURE = 0.5
MODEL_PATH = "../models/"

def plot_latent_space(dataset, network, device, dark_background=False, dimensions=3):
    """
    Plots the images of the test set projected onto the latent space
    (through the encoder of the BiGAN), with their respective labels for
    more clarity.

    Note that if the dimension of the latent space is > 2, T-SNE is used
    for the visualization (reducing the dimension of the data)
    """
    if dataset is None:
        return

    z_array = np.zeros((
        len(dataset.dataset),
        network.get_z_dim()))
    labels = np.zeros(len(dataset.dataset))
    batch_id = 0
    for x, label in dataset:
        x = x.to(device)
        z1 = network.get_latent_space(x)
        np_z = z1.cpu().detach().numpy()
        np_label = label.detach().numpy()
        batch_size = np_z.shape[0]
        z_array[batch_id:batch_id + batch_size] = np_z[:, :]
        labels[batch_id:batch_id + batch_size] = np_label
        batch_id += batch_size

    if dark_background:
        plt.style.use('dark_background')

    fig = plt.figure(figsize=(12, 10))

    if network.get_z_dim() > dimensions:
        z_array = TSNE(n_components=dimensions).fit_transform(z_array)

    if dimensions == 2:
        plt.scatter(z_array[:, 0], z_array[:, 1], c=labels)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.colorbar()

    elif dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')
        cloud = ax.scatter(
            z_array[:, 0],
            z_array[:, 1],
            z_array[:, 2],
            c=labels)

        ax.set_xlabel("z[0]")
        ax.set_ylabel("z[1]")
        ax.set_zlabel("z[2]")
        fig.colorbar(cloud)

    else:
        print("The dimensions param should be set to 2 or 3")
        return

    plt.title("Latent space")

    if not os.path.exists("../results"):
        os.mkdir("../results")

    plt.savefig("../results/latent_space.png")
    plt.show()