import os
import logging
from collections import OrderedDict
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel
from models.modules.loss import GANLoss
import models.modules.block as B
torch.autograd.set_detect_anomaly(True)
logger = logging.getLogger('base')



class NTIRE_model(BaseModel):
    def __init__(self, opt):
        super(NTIRE_model, self).__init__(opt)
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G1(opt).to(self.device)  # G1
        if self.is_train:
            self.netD = networks.define_D2(opt=opt)
            self.netD = self.netD.to(self.device)
            self.netQ = networks.define_Q(opt).to(self.device)
            self.netG.train()
            self.netD.train()
        self.load()  # load G and D if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None
            self.weight_kl = 1e-2
            self.weight_D = 1e-3
            self.l_gan_w = train_opt['gan_weight']
            self.l_pix_w = train_opt['pixel_weight']
            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False,Rlu=True).to(self.device) #Rlu=True if feature taken before relu, else false
                self.percepLossInfo = []
                
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)

            #D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'], \
                weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
            self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
        # print network
        self.print_network()

    def feed_data(self, data, need_HR=True):
        # LR
        self.var_L = data['LR'].to(self.device)
        if need_HR:  # train or val
            self.var_H = data['HR'].to(self.device)
    
    def calculatePercepLoss(self, HRFeatures, SRFeatures):
        loss = 0.0
        divFactor = [[224,64],[112,128],[56,256],[28,512]]
        self.percepLossInfo = []
        for i in range(len(HRFeatures)):
            scaledHRFeatures = torch.div(HRFeatures[i], divFactor[i][0] * divFactor[i][0] * divFactor[i][1])
            scaledSRFeatures = torch.div(SRFeatures[i], divFactor[i][0] * divFactor[i][0] * divFactor[i][1])
            l2norm = self.cri_fea(scaledHRFeatures, scaledSRFeatures)
            self.percepLossInfo.insert(i, l2norm)
            loss += l2norm
        return loss
    
    def optimize_parameters(self, step):
        # G
        n1 = torch.nn.Upsample(scale_factor=1/4)
        for i in range(20):
            self.optimizer_G.zero_grad()
            self.SR = self.netG(n1(self.var_L))
            Quality_loss = 2e-7 * (5-torch.mean(self.netQ(self.SR).detach()))
            #Quality_loss = 0
            self.HR_D = self.netD(self.var_H.detach())
            self.LR_D = self.netD(self.var_L.detach())
            self.SR_D = self.netD(self.SR)

            l_g_total = 0
            
            if(self.l_pix_w > 0):
               l_g_pix = self.l_pix_w * self.cri_pix(self.SR, self.var_H)
               l_g_total += l_g_pix

            l_g_percep = self.l_fea_w * self.calculatePercepLoss(self.netF(self.var_H.detach()), self.netF(self.SR.detach()))
            l_g_total += l_g_percep

            l_g_dis = self.l_gan_w * self.cri_gan((self.SR_D, self.HR_D, self.LR_D), 0) #gen = (1-SR_D)^2
            l_g_total += 2*l_g_dis

            l_g_total += Quality_loss
            l_g_total =  l_g_total

            l_g_total.backward()
            self.optimizer_G.step()

        
        self.optimizer_D.zero_grad()
        self.HR_D = self.netD(self.var_H.detach())
        self.LR_D = self.netD(self.var_L.detach())
        self.SR_D = self.netD(self.SR.detach())
    
        
        l_d_total = 0

        l_d_total = self.l_gan_w * self.cri_gan((self.SR_D, self.LR_D, self.HR_D), 1)
        l_d_total = 1*l_d_total

        l_d_total.backward()
        self.optimizer_D.step()

    

        # set log
        self.log_dict['l_g_percep_output1'] = self.percepLossInfo[0].item()
        self.log_dict['l_g_percep_output2'] = self.percepLossInfo[1].item()
        self.log_dict['l_g_percep_output3'] = self.percepLossInfo[2].item()
        self.log_dict['l_g_percep_output4'] = self.percepLossInfo[3].item()
        self.log_dict['l_g_percep'] = l_g_percep.item()
        self.log_dict['l_g_d'] = l_g_dis.item()
        if(self.l_pix_w > 0):
            self.log_dict['l_g_pix'] = l_g_pix.item()
        self.log_dict['l_d_total'] = l_d_total.item()
        if(Quality_loss > 0):
            self.log_dict['Quality_Loss'] = Quality_loss.item()
    
    def test(self):
        self.netG.eval()

        with torch.no_grad():
            n1 = torch.nn.Upsample(scale_factor=1/4)
            self.SR = self.netG(n1(self.var_L))
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['SR'] = self.SR.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
        
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading pretrained model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD)
        

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
