from DCGAN.DCGAN_main import DCGAN_main
from GAN.GAN_main import GAN_main
from WGAN_GP.WGAN_GP_main import WGAN_GP_main
from WGAN.WGAN_main import WGAN_main
from SN_DCGAN.SN_DCGAN_main import SN_DCGAN_main
from option import option

opt = option().parse()

if opt.type == 'GAN':
    GAN_main()
elif opt.type == 'DCGAN':
    DCGAN_main()
elif opt.type == 'WGAN_GP':
    WGAN_GP_main()
elif opt.type == 'WGAN':
    WGAN_main()
else:
    SN_DCGAN_main()