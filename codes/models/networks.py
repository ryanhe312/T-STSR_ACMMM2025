import models.modules.Sakuya_arch as Sakuya_arch
import models.modules.Sakuya_arch_microscopy as Sakuya_arch_microscopy
import models.modules.Sakuya_arch_microscopy_mamba as Sakuya_arch_microscopy_mamba

####################
# define network
####################
# Generator


def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'LunaTokis':
        netG = Sakuya_arch.LunaTokis(nf=opt_net['nf'], nframes=opt_net['nframes'],
                                     groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                                     back_RBs=opt_net['back_RBs'])
    elif which_model == 'LunaTokis_Microscopy':
        netG = Sakuya_arch_microscopy.LunaTokis(nf=opt_net['nf'], nframes=opt_net['nframes'],
                                     groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                                     back_RBs=opt_net['back_RBs'])
    elif which_model == 'LunaTokis_Microscopy_Mamba':
        netG = Sakuya_arch_microscopy_mamba.LunaTokis(nf=opt_net['nf'], nframes=opt_net['nframes'],
                                     groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                                     back_RBs=opt_net['back_RBs'], norm=opt_net['norm'], vssm=opt_net['vssm'])
    else:
        raise NotImplementedError(
            'Generator model [{:s}] not recognized'.format(which_model))

    return netG
