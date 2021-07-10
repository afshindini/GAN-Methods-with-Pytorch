# import libraries
import matplotlib.pyplot as plt
import os
import itertools
import torch
from torch import nn


# Function for creation of noise vector as the input of the Generator
def noise_vector(n_sample, z_dim, device='cpu'):
    return torch.randn(n_sample, z_dim, device=device)


# This function saves the DCGAN images in a grid format
def save_result(folder_name, image_no, z_dim, generator, num_epoch, device):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    path = folder_name + '/' + str(num_epoch) + '.png'
    noise = noise_vector(image_no * image_no, z_dim, device=device)
    generator.eval()
    test_images = generator(noise)
    generator.train()
    fig, ax = plt.subplots(image_no, image_no, figsize=(5, 5))
    for i, j in itertools.product(range(image_no), range(image_no)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(image_no * image_no):
        i = k // image_no
        j = k % image_no
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')
    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)
    plt.close()


# This function plot the generator and loss function
def plot_loss(folder_name, epochs, d_losses, g_losses):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    path = folder_name + '/losses' + '.png'
    x = [i + 1 for i in range(epochs)]
    plt.plot(x, d_losses, label="Discriminator loss")
    plt.plot(x, g_losses, label="Generator loss")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(path)
    plt.close()


# This function saves the GAN images in a grid format
def save_result_GAN(folder_name, image_no, z_dim, generator, num_epoch, device, size=(28, 28)):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    path = folder_name + '/' + str(num_epoch) + '.png'
    noise = noise_vector(image_no * image_no, z_dim, device=device)
    generator.eval()
    test_images = generator(noise)
    generator.train()
    fig, ax = plt.subplots(image_no, image_no, figsize=(5, 5))
    for i, j in itertools.product(range(image_no), range(image_no)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(image_no * image_no):
        i = k // image_no
        j = k % image_no
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, :].cpu().view(-1, *size).squeeze().data.numpy(), cmap='gray')
    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)
    plt.close()


# This function is used for initializing the weights of discriminator and generator weights
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
