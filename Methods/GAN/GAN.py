from torch import nn
import torch


# Define Noise vector as the input of the Generator
def noise_vector(n_sample, z_dim, device='cpu'):
    return torch.randn(n_sample, z_dim, device=device)


# Define Generator Network Architecture
class Generator(nn.Module):
    def __init__(self, z_dim=64, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.gen_block(z_dim, hidden_dim),
            self.gen_block(hidden_dim, hidden_dim * 2),
            self.gen_block(hidden_dim * 2, hidden_dim * 4),
            self.gen_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid())

    # Build the neural block used in generator network
    def gen_block(self, input_dim, output_dim):
        return nn.Sequential(nn.Linear(input_dim, output_dim),
                             nn.BatchNorm1d(output_dim),
                             nn.ReLU(inplace=True))

    def forward(self, x):
        output = self.gen(x)
        return output


# Define Discriminator Network Architecture
class Discriminator(nn.Module):
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(self.disc_block(im_dim, hidden_dim * 4),
                                  self.disc_block(hidden_dim * 4, hidden_dim * 2),
                                  self.disc_block(hidden_dim * 2, hidden_dim),
                                  nn.Linear(hidden_dim, 1))

    # Build the neural block used in discriminator network
    def disc_block(self, input_dim, output_dim):
        return nn.Sequential(nn.Linear(input_dim, output_dim),
                             nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        output = self.disc(x)
        return output
