from option import option
from Methods.GAN.GAN import Generator, Discriminator
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch import nn
import torch
from Utility.utility import save_result_GAN, plot_loss, noise_vector
from tqdm import tqdm
import time


def GAN_main():
    # Initial Values
    opt = option().parse()
    criterion = nn.BCEWithLogitsLoss()

    # Load and preprocessing Data
    transform = transforms.Compose([
        transforms.ToTensor()])
    dataloader = DataLoader(
        MNIST('.', download=True, transform=transform),
        batch_size=opt.batch,
        shuffle=True)

    # Define generator and discriminator models and optimizers and move the parameters to the device
    gen = Generator(opt.zdim, opt.img_dim, opt.hidden_dim).to(opt.device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    disc = Discriminator(opt.img_dim, opt.hidden_dim).to(opt.device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

    # These parameters are used to record the loss of each epoch for final plotting
    plot_d_losses = []
    plot_g_losses = []

    # Training Process
    print("[Training starts:...]")
    for epoch in range(opt.epochs):

        # These losses used for calculating the loss of each epoch by taking the mean of losses of batches in one epoch
        D_losses = []
        G_losses = []
        epoch_start_time = time.time()  # Used for recording the elapsed time for each epoch

        # Dataloader returns the batches
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)      # Get the length of each batch
            real = real.view(cur_batch_size, -1).to(opt.device)     # ravel the input to be compatible with network

            # Calculate discriminator loss and update related weights
            disc_opt.zero_grad()
            # Create the fake image and calculate discriminator loss for fake images
            fake_noise = noise_vector(cur_batch_size, opt.zdim, device=opt.device)
            fake = gen(fake_noise)
            disc_fake_pred = disc(fake.detach())    # to update only discriminator's weights, detach() shall be used
            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            # calculate discriminator loss for real images
            disc_real_pred = disc(real)
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            # Calculate the total discriminator loss
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            # Keep track of the discriminator loss in each batch
            D_losses.append(disc_loss)
            # Update gradients for discriminator
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            # Calculate generator loss and update related weights
            gen_opt.zero_grad()
            # Create the fake image and calculate generator loss
            fake_noise_2 = noise_vector(cur_batch_size, opt.zdim, device=opt.device)
            fake_2 = gen(fake_noise_2)
            disc_fake_pred = disc(fake_2)
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            # Update gradients for generator
            gen_loss.backward()
            gen_opt.step()
            # Keep track of the generator loss in each batch
            G_losses.append(gen_loss)

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        print('[%d/%d] - passed time: %.2f, loss_d: %.3f, loss_g: %.3f' % (
            epoch, opt.epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
            torch.mean(torch.FloatTensor(G_losses))))
        save_result_GAN("results", 5, opt.zdim, gen, epoch, opt.device)     # Save created images
        plot_d_losses.append(torch.mean(torch.FloatTensor(D_losses)))
        plot_g_losses.append(torch.mean(torch.FloatTensor(G_losses)))
    plot_loss('loss', opt.epochs, plot_d_losses, plot_g_losses)         # Save the final loss diagram
