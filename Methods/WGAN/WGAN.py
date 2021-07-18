from torch import nn
import torch


# Calculate generator loss by finding mean value of critic for fake images
def generator_loss(critic_fake_pred):
    return -torch.mean(critic_fake_pred)


# Calculate critic loss considering gradient penalty
def critic_loss(critic_fake_pred, critic_real_pred):
    return -torch.mean(critic_real_pred) + torch.mean(critic_fake_pred)


# Define Generator Network Architecture
class Generator(nn.Module):
    def __init__(self, z_dim, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.gen_block(z_dim, hidden_dim * 4),
            self.gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.gen_block(hidden_dim * 2, hidden_dim),
            self.gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True))

    def gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                                 nn.BatchNorm2d(output_channels),
                                 nn.ReLU(inplace=True))
        else:
            return nn.Sequential(nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                                 nn.Tanh())

    def forward(self, x):
        x = x.view(len(x), self.z_dim, 1, 1)
        output = self.gen(x)
        return output


# Define Discriminator Network Architecture
class Critic(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(self.critic_block(im_chan, hidden_dim),
                                  self.critic_block(hidden_dim, hidden_dim * 2),
                                  self.critic_block(hidden_dim * 2, 1, final_layer=True))

    def critic_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        # Build the neural block
        if not final_layer:
            return nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                                 nn.BatchNorm2d(output_channels),
                                 nn.LeakyReLU(0.2, inplace=True))
        else:
            return nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size, stride))

    def forward(self, x):
        output = self.disc(x)
        return output.view(len(output), -1)
