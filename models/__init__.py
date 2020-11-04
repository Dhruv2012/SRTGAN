import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'sr':
        from .SR_model import SRModel as M
    elif model == 'srgan':
        from .SRGAN_model import SRGANModel as M
    elif model == 'srragan':
        from .SRRaGAN_model import SRRaGANModel as M
    elif model == 'sftgan':
        from .SFTGAN_ACD_model import SFTGAN_ACD_Model as M
    elif model == 'srfeat':
        from .SRFeatModel import SRFeatModel as M
    elif model == 'bicyclegan':
        from .DualGAN import DualGAN as M
    elif model == 'ntire':
        from .NTIRE_model import NTIRE_model as M
    elif model == 'icpr':
        from .ICPR_model import ICPR_model as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
