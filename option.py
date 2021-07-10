import argparse


class option:
    def __init__(self):
        self.parser = argparse.ArgumentParser(prog="main.py",
                                              description="List of the possible options",
                                              epilog="The main code starts running using parsed data",
                                              allow_abbrev=False)

        self.parser.add_argument('--type', default='WGAN', help='Define the type of GAN network')
        self.parser.add_argument('--zdim', type=int, default=64, help='dimension of latent space')
        self.parser.add_argument('--batch', type=int, default=128, help='batch size')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
        self.parser.add_argument('--device', default='cuda', help='device for running the code')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='the beta1 parameter of optimization')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='the beta2 parameter of optimization')
        self.parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
        self.parser.add_argument('--image_resize', type=int, default=64, help='image resize shape')
        self.parser.add_argument('--hidden_dim', type=int, default=64, help='No. of hidden layer for the first layer')
        self.parser.add_argument('--img_channel', type=int, default=1, help='number of image channels')
        self.parser.add_argument('--img_dim', type=int, default=784, help='number of image dimensions in GAN')
        self.parser.add_argument('--display_freq', type=int, default=500, help='visualization frequency display')
        self.parser.add_argument('--critic_repeats', type=int, default=5, help='number of repeats for credit')
        self.parser.add_argument('--c_lambda', type=float, default=10, help='Gradient penalty factor as lambda')
        self.opt = None

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
