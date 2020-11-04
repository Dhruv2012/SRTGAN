import os
import logging
from collections import OrderedDict
import itertools
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel
from models.modules.loss import GANLoss

logger = logging.getLogger('base')


class DualGAN(BaseModel):
    def __init__(self, opt):
        super(DualGAN, self).__init__(opt)
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG1 = networks.define_G1(opt).to(self.device)  # G1
        if self.is_train:
            self.netG2 = networks.define_G2(opt).to(self.device)  # G2
            self.netD1 = networks.define_D(opt).to(self.device)  # D
            self.netD2 = networks.define_D(opt).to(self.device)  # D
            self.netQ = networks.define_Q(opt).to(self.device)
            self.netG1.train()
            self.netG2.train()
            self.netD1.train()
            self.netD2.train()
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
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

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
                self.netF = networks.define_F(opt, use_bn=False,Rlu=True).to(self.device)   #Rlu=True if feature taken before relu, else false

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters are for WGAN
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            
            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG1.parameters(), self.netG2.parameters()),lr=train_opt['lr_G'], \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD1.parameters(), self.netD2.parameters()),lr=train_opt['lr_G'], \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
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


    def optimize_parameters(self, step):
        # G

        self.optimizer_G.zero_grad()
        self.SR = self.netG1(self.var_L)
        self.LR_est = self.netG2(self.SR)
        l_g_total1 = 0
        
        Quality_loss = 1e-6 * (5-torch.mean(self.netQ(self.SR).detach()))
        l_g_total1 += Quality_loss

        l_g_pix1 = self.l_pix_w * self.cri_pix(self.LR_est, self.var_L)
        l_g_total1 += l_g_pix1
        if self.cri_fea:  # feature loss
            real_fea = self.netF(self.var_L)
            fake_fea = self.netF(self.LR_est)
            l_g_fea1 = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
            l_g_total1 += l_g_fea1
        # G gan + cls loss

        pred_g_fake = self.netD1(self.LR_est)
        l_g_gan1 = self.l_gan_w * self.cri_gan(pred_g_fake, True)
        l_g_total1 += l_g_gan1
        
        self.DR = self.netG2(self.var_H)
        self.HR_est = self.netG1(self.DR)

        l_g_total2 = 0

        l_g_pix2 = self.l_pix_w * self.cri_pix(self.HR_est, self.var_H)
        l_g_total2 += l_g_pix2
        if self.cri_fea:  # feature loss
            real_fea = self.netF(self.var_H)
            fake_fea = self.netF(self.HR_est)
            l_g_fea2 = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
            l_g_total2 += l_g_fea2
        # G gan + cls loss
        pred_g_fake = self.netD2(self.HR_est)
        l_g_gan2 = self.l_gan_w * self.cri_gan(pred_g_fake, True)
        l_g_total2 += l_g_gan2

        total_lossG = l_g_total1 + l_g_total2
        total_lossG.backward()

        self.optimizer_G.step()

        # D
        self.optimizer_D.zero_grad()
        # real data
        pred_d_real1 = self.netD1(self.var_L)
        l_d_real1 = self.cri_gan(pred_d_real1, True)
        # fake data
        pred_d_fake1 = self.netD1(self.LR_est.detach())  # detach to avoid BP to G
        l_d_fake1 = self.cri_gan(pred_d_fake1, False)
        l_d_total1 = l_d_real1 + l_d_fake1

        pred_d_real2 = self.netD2(self.var_H)
        l_d_real2 = self.cri_gan(pred_d_real2, True)
        # fake data
        pred_d_fake2 = self.netD2(self.HR_est.detach())  # detach to avoid BP to G
        l_d_fake2 = self.cri_gan(pred_d_fake2, False)

        l_d_total2 = l_d_real2 + l_d_fake2
        l_d_total = l_d_total1 + l_d_total2
        
        l_d_total.backward()
        self.optimizer_D.step()

        # set log
        
        self.log_dict['l_g_total1'] = l_g_total1.item()
        self.log_dict['l_g_total2'] = l_g_total2.item()
        
        self.log_dict['l_d_total1'] = l_d_total1.item()
        self.log_dict['l_d_total2'] = l_d_total2.item()

    def test(self):
        self.netG1.eval()
        with torch.no_grad():
            self.SR = self.netG1(self.var_L)
        self.netG1.train()

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
        s, n = self.get_network_description(self.netG1)
        if isinstance(self.netG1, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG1.__class__.__name__,
                                             self.netG1.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG1.__class__.__name__)
        logger.info('Network G1 structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

        s, n = self.get_network_description(self.netG2)
        if isinstance(self.netG2, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG2.__class__.__name__,
                                             self.netG2.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG2.__class__.__name__)
        logger.info('Network G2 structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD1)
            if isinstance(self.netD1, nn.DataParallel):
                net_struc_str = '{} - {}'.format(self.netD1.__class__.__name__,
                                                self.netD1.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD1.__class__.__name__)
            logger.info('Network D1 structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

            s, n = self.get_network_description(self.netD2)
            if isinstance(self.netD2, nn.DataParallel):
                net_struc_str = '{} - {}'.format(self.netD2.__class__.__name__,
                                                self.netD2.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD2.__class__.__name__)
            logger.info('Network D2 structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                    self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)
                logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

    def load(self):
        load_path_G1 = self.opt['path']['pretrain_model_G1']
        if load_path_G1 is not None:
            logger.info('Loading pretrained model for G1 [{:s}] ...'.format(load_path_G1))
            self.load_network(load_path_G1, self.netG1)
        load_path_G2 = self.opt['path']['pretrain_model_G2']
        if load_path_G2 is not None:
            logger.info('Loading pretrained model for G2 [{:s}] ...'.format(load_path_G2))
            self.load_network(load_path_G2, self.netG2)
        load_path_D1 = self.opt['path']['pretrain_model_D1']
        if self.opt['is_train'] and load_path_D1 is not None:
            logger.info('Loading pretrained model for D1 [{:s}] ...'.format(load_path_D1))
            self.load_network(load_path_D1, self.netD1)
        load_path_D2 = self.opt['path']['pretrain_model_D2']
        if self.opt['is_train'] and load_path_D2 is not None:
            logger.info('Loading pretrained model for D2 [{:s}] ...'.format(load_path_D2))
            self.load_network(load_path_D2, self.netD2)

        load_path_Q = "/media/ml/Data Disk/Kalpesh/NTIRE/latest_G.pth"
        logger.info('Loading pretrained model for Q [{:s}] ...'.format(load_path_Q))
        self.load_network(load_path_Q, self.netQ)

    def save(self, iter_step):
        self.save_network(self.netG1, 'G1', iter_step)
        self.save_network(self.netD1, 'D1', iter_step)
        self.save_network(self.netG2, 'G2', iter_step)
        self.save_network(self.netD2, 'D2', iter_step)
