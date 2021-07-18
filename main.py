from Methods.DCGAN.DCGAN_main import DCGAN_main
from Methods.GAN.GAN_main import GAN_main
from Methods.WGAN_GP.WGAN_GP_main import WGAN_GP_main
from Methods.WGAN.WGAN_main import WGAN_main
from Methods.SN_DCGAN.SN_DCGAN_main import SN_DCGAN_main
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
elif opt.type == 'SN_DCGAN':
    SN_DCGAN_main()
else:
    print("The type is not correct")