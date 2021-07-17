from option import option
from WGAN_GP.WGAN_GP import Generator, Critic, gradient_penalty, generator_loss, critic_loss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import torch
from Utility.utility import save_result_GAN, plot_loss, weights_init, noise_vector
from tqdm import tqdm
import time


def WGAN_GP_main():
    # Initial Values
    opt = option().parse()

    # Preprocessing Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    dataloader = DataLoader(
        MNIST('.', download=True, transform=transform),
        batch_size=opt.batch,
        shuffle=True)

    # Define generator and critic models and move the parameters to the device
    gen = Generator(opt.zdim, opt.img_channel, opt.hidden_dim).to(opt.device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    crit = Critic(opt.img_channel, opt.hidden_dim).to(opt.device)
    crit_opt = torch.optim.Adam(crit.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

    # # Initial weights for generator and critic models
    gen = gen.apply(weights_init)
    crit = crit.apply(weights_init)

    # These parameters are used to record the loss of each epoch for final plotting
    plot_critic_losses = []
    plot_g_losses = []

    # Training Process
    print("[Training starts:...]")
    for epoch in range(opt.epochs):

        # These losses used for calculating the loss of each epoch by taking the mean of losses of batches in one epoch
        crit_losses = []
        G_losses = []
        epoch_start_time = time.time()  # Used for recording the elapsed time for each epoch

        # Dataloader returns the batches
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.to(opt.device)

            repeat_critic_loss = 0
            # Update critic
            for _ in range(opt.critic_repeats):
                crit_opt.zero_grad()
                fake_noise = noise_vector(cur_batch_size, opt.zdim, device=opt.device)
                fake = gen(fake_noise)
                crit_fake_pred = crit(fake.detach())
                crit_real_pred = crit(real)

                epsilon = torch.rand(len(real), 1, 1, 1, device=opt.device, requires_grad=True)
                gp = gradient_penalty(crit, real, fake.detach(), epsilon)
                crit_loss = critic_loss(crit_fake_pred, crit_real_pred, gp, opt.c_lambda)

                # Keep track of the average critic loss
                repeat_critic_loss += crit_loss.item()/opt.critic_repeats

                # Update gradients
                crit_loss.backward(retain_graph=True)
                # Update optimizer
                crit_opt.step()

            crit_losses.append(repeat_critic_loss)

            # Update generator
            gen_opt.zero_grad()
            fake_noise_2 = noise_vector(cur_batch_size, opt.zdim, device=opt.device)
            fake_2 = gen(fake_noise_2)
            crit_fake_pred_2 = crit(fake_2)
            gen_loss = generator_loss(crit_fake_pred_2)
            gen_loss.backward()
            gen_opt.step()

            # Keep track of the average generator loss
            G_losses.append(gen_loss)

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        print('[%d/%d] - passed time: %.2f, loss_d: %.3f, loss_g: %.3f' % (
            epoch, opt.epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(crit_losses)),
            torch.mean(torch.FloatTensor(G_losses))))
        save_result_GAN("results", 5, opt.zdim, gen, epoch, opt.device)
        plot_critic_losses.append(torch.mean(torch.FloatTensor(crit_losses)))
        plot_g_losses.append(torch.mean(torch.FloatTensor(G_losses)))
    plot_loss('loss', opt.epochs, plot_critic_losses, plot_g_losses)
