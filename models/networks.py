import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init

import models.modules.architecture as arch
import models.modules.sft_arch as sft_arch
logger = logging.getLogger('base')
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Generator
def define_G1(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_G']

    netG = arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
        nb=opt_net['nb'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'],
        act_type='leakyrelu', mode=opt_net['mode'], upsample_mode='upconv')
    
    if opt['is_train']:
        init_weights(netG, init_type='kaiming', scale=0.1)
    if gpu_ids:
        print(torch.cuda.is_available())
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG

def define_G3(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_G']

    netG = arch.RRDBNet2(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
        nb=opt_net['nb'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'],
        act_type='leakyrelu', mode=opt_net['mode'], upsample_mode='upconv')
    
    if opt['is_train']:
        init_weights(netG, init_type='kaiming', scale=0.1)
    if gpu_ids:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG

def define_G2(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_G']

    netG = arch.DegNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
        nb=opt_net['nb'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'],
        act_type='leakyrelu', mode=opt_net['mode'], upsample_mode='upconv')
    
    if opt['is_train']:
        init_weights(netG, init_type='kaiming', scale=0.1)
    if gpu_ids:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG

# Discriminator
def define_D(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_D']
    
    netD = arch.Discriminator(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
        norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'],out_feat=256)
    
    init_weights(netD, init_type='kaiming', scale=1)
    if gpu_ids:
        netD = nn.DataParallel(netD)
    return netD
def define_D2(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_D']
    
    netD = arch.Discriminator(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
        norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'],out_feat=1)
    
    init_weights(netD, init_type='kaiming', scale=1)
    if gpu_ids:
        netD = nn.DataParallel(netD)
    return netD

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_patchD(opt, netD="basic", n_layers_D=3):
    """Create a discriminator
    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a discriminator
    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.
        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)
        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_D']
    net = None
    norm_layer = get_norm_layer(norm_type=opt_net['norm_type'])

    if netD == 'basic':  # default PatchGAN classifier
        net = arch.NLayerDiscriminator(input_nc = opt_net['in_nc'], ndf=opt_net['nf'], n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = arch.NLayerDiscriminator(input_nc = opt_net['in_nc'], ndf=opt_net['nf'], n_layers= n_layers_D, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    
    init_weights(net, init_type='kaiming', scale=1)
    if gpu_ids:
        net = nn.DataParallel(net)
    return net
    



def define_F(opt, use_bn=False, Rlu=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # pytorch pretrained VGG19-54, before ReLU.
    if not Rlu:
        if use_bn:
            feature_layer = 49
        else:
            feature_layer = 34
    else:
        if use_bn:
            feature_layer = 51
        else:
            feature_layer = 35

    netF = arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, \
        use_input_norm=True, device=device)
    # netF = arch.ResNet101FeatureExtractor(use_input_norm=True, device=device)
    if gpu_ids:
        netF = nn.DataParallel(netF)
    netF.eval()  # No need to train
    return netF

def define_Q(opt):
    gpu_ids = opt['gpu_ids']

    netQ = arch.VGGGAPQualifierModel(in_nc=opt['in_nc'], nf=opt['nf'],height=opt['height'],width=opt['width'])
    if gpu_ids:
        netQ = nn.DataParallel(netQ)
    return netQ

def define_Q2():
    # gpu_ids = opt['gpu_ids']

    netQ2 = arch.VGGGAPQualifierwaveModel()
    if False:
        netQ2 = nn.DataParallel(netQ2)
    return netQ2