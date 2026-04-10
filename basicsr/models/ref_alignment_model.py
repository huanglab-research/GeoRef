import os
import importlib
import logging
import os.path as osp
from collections import OrderedDict
from tqdm import tqdm
import time
import datetime
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
from basicsr.archs import build_network
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import ProgressBar, get_root_logger, imwrite, tensor2img
import basicsr.utils.metrics as metrics
from .base_model import BaseModel

from basicsr.utils.registry import MODEL_REGISTRY

loss_module = importlib.import_module('basicsr.models.losses')


@MODEL_REGISTRY.register()
class RefAlignmentModel(BaseModel):

    def __init__(self, opt):
        super(RefAlignmentModel, self).__init__(opt)
        self.opt = opt

        # net_map does not have any trainable parameters.
        self.net_map = build_network(opt['network_map'])
        self.net_map = self.model_to_device(self.net_map)

        # define network for feature extraction
        self.net_extractor = build_network(opt['network_extractor'])
        self.net_extractor = self.model_to_device(self.net_extractor)
        # self.print_network(self.net_extractor)
        # load pretrained feature extractor
        load_path = self.opt['path'].get('pretrain_network_feature_extractor',
                                         None)
        if load_path is not None:
            self.load_network(self.net_extractor, load_path,
                              self.opt['path']['strict_load'])
            for param in self.net_extractor.parameters():
                param.requires_grad = False

        # define network for rrdb
        self.net_rrdb = build_network(opt['network_rrdb'])
        self.net_rrdb = self.model_to_device(self.net_rrdb)
        # self.print_network(self.net_rrdb)
        # load pretrained rrdb
        load_path = self.opt['path'].get('pretrain_network_rrdb', None)
        if load_path is not None:
            self.load_network(self.net_rrdb, load_path,
                                   self.opt['path']['strict_load'])
            for param in self.net_rrdb.parameters():
                param.requires_grad = False

        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path']['strict_load'])
        if self.is_train:
            self.net_g.train()

            logger = get_root_logger()
            # optimizers
            train_opt = self.opt['train']
            weight_decay_g = train_opt.get('weight_decay_g', 0)
            optim_params_g = []
            optim_params_offset = []
            optim_params_relu2_offset = []
            optim_params_relu3_offset = []
            if train_opt.get('lr_relu3_offset', None):
                optim_params_relu3_offset = []
            for name, v in self.net_g.named_parameters():
                if v.requires_grad:
                    if 'offset' in name:
                        if 'small' in name:
                            logger.info(name)
                            optim_params_relu3_offset.append(v)
                        elif 'medium' in name:
                            logger.info(name)
                            optim_params_relu2_offset.append(v)
                        else:
                            optim_params_offset.append(v)
                    else:
                        optim_params_g.append(v)

            self.optimizer_g = torch.optim.Adam(
                [{
                    'params': optim_params_g
                }, {
                    'params': optim_params_offset,
                    'lr': train_opt['lr_offset']
                }, {
                    'params': optim_params_relu3_offset,
                    'lr': train_opt['lr_relu3_offset']
                }, {
                    'params': optim_params_relu2_offset,
                    'lr': train_opt['lr_relu2_offset']
                }],
                lr=train_opt['lr_g'],
                weight_decay=weight_decay_g,
                betas=train_opt['beta_g'])

            self.optimizers.append(self.optimizer_g)
            self.init_training_settings()


    def init_training_settings(self):
        train_opt = self.opt['train']

        logger = get_root_logger()
        if self.opt.get('network_d', None):
            # define network net_d
            self.net_d = build_network(self.opt['network_d'])
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)
            # load pretrained models
            load_path = self.opt['path'].get('pretrain_network_d', None)
            if load_path is not None:
                self.load_network(self.net_d, load_path,
                                  self.opt['path']['strict_load'])
        else:
            logger.info('No discriminator.')
            self.net_d = None

        if self.net_d:
            self.net_d.train()

        # define losses
        if train_opt['pixel_weight'] > 0:
            cri_pix_cls = getattr(loss_module, train_opt['pixel_criterion'])
            self.cri_pix = cri_pix_cls(
                loss_weight=train_opt['pixel_weight'],
                reduction='mean').to(self.device)
        else:
            logger.info('Remove pixel loss.')
            self.cri_pix = None

        if train_opt.get('perceptual_opt', None):
            cri_perceptual_cls = getattr(loss_module, 'PerceptualLoss')
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            logger.info('Remove perceptual loss.')
            self.cri_perceptual = None

        if train_opt.get('style_opt', None):
            cri_style_cls = getattr(loss_module, 'PerceptualLoss')
            self.cri_style = cri_style_cls(**train_opt['style_opt']).to(
                self.device)
        else:
            logger.info('Remove style loss.')
            self.cri_style = None

        if train_opt.get('texture_opt', None):
            cri_texture_cls = getattr(loss_module, 'TextureLoss')
            self.cri_texture = cri_texture_cls(**train_opt['texture_opt']).to(
                self.device)
        else:
            logger.info('Remove texture loss.')
            self.cri_texture = None

        if train_opt.get('gan_type', None):
            cri_gan_cls = getattr(loss_module, 'GANLoss')
            self.cri_gan = cri_gan_cls(
                train_opt['gan_type'],
                real_label_val=1.0,
                fake_label_val=0.0,
                loss_weight=train_opt['gan_weight']).to(self.device)

            if train_opt['grad_penalty_weight'] > 0:
                cri_grad_penalty_cls = getattr(loss_module,
                                               'GradientPenaltyLoss')
                self.cri_grad_penalty = cri_grad_penalty_cls(
                    loss_weight=train_opt['grad_penalty_weight']).to(
                        self.device)
            else:
                logger.info('Remove gradient penalty.')
                self.cri_grad_penalty = None
        else:
            logger.info('Remove GAN loss.')
            self.cri_gan = None

        # we need to train the net_g with only pixel loss for several steps
        self.net_g_pretrain_steps = train_opt['net_g_pretrain_steps']
        self.net_d_steps = train_opt.get('net_d_steps', 1)
        self.net_d_init_steps = train_opt.get('net_d_init_steps', 0)

        # optimizers
        if self.net_d:
            weight_decay_d = train_opt.get('weight_decay_d', 0)
            self.optimizer_d = torch.optim.Adam(
                self.net_d.parameters(),
                lr=train_opt['lr_d'],
                weight_decay=weight_decay_d,
                betas=train_opt['beta_d'])
            self.optimizers.append(self.optimizer_d)

        self.margin = train_opt['margin']
        self.is_contras = train_opt['add_contras']
        self.contras_weight = train_opt['contras_weight']

        # check the schedulers
        self.setup_schedulers()

        self.log_dict = OrderedDict()

    def feed_data(self, data, train=True):
        self.img_in_lq = data['img_in_lq'].to(self.device)
        self.img_ref = data['img_ref'].to(self.device)
        self.gt = data['img_in'].to(self.device)  # gt
        self.match_img_in = data['img_in_up'].to(self.device)

    def optimize_parameters(self, step):
        self.sife, self.sisr_in = self.net_rrdb(self.img_in_lq)
        self.features = self.net_extractor(self.match_img_in, self.img_ref)
        self.pre_flow, self.pre_offset, self.pre_sim, self.img_ref_feat = self.net_map(
                self.features, self.img_ref)
        self.output, self.mid_ref_feat = self.net_g(self.sisr_in, self.sife, self.pre_offset[0], self.pre_flow[0], None, self.img_ref_feat)
        if self.is_contras:
            self.features_p = self.net_extractor(self.match_img_in, self.gt)
            self.pre_flow_p, self.pre_offset_p, self.pre_sim_p, self.img_gt_feat = self.net_map(self.features_p, self.gt)
            self.pre_flow_n, self.pre_offset_n, self.pre_sim_n, self.img_up_feat = self.net_map(self.features_p, self.match_img_in)

            self.output_p, self.mid_ref_feat_p = self.net_g(self.sisr, self.sife, self.pre_offset_p[0], self.pre_flow_p[0], None, self.img_gt_feat)
            self.output_n, self.mid_ref_feat_n = self.net_g(self.sisr, self.sife, self.pre_offset_p[0], self.pre_flow_p[0], None, self.img_up_feat)



        self.optimizer_g.zero_grad()
        l_g_total = 0
        l_pix = self.cri_pix(self.output, self.gt)
        # set log
        self.log_dict['l_pix'] = l_pix.item()
        l_g_total += l_pix

        contras_loss, pos_dist, neg_dist = self.contras_loss()
        if torch.isnan(contras_loss) or torch.isinf(contras_loss):
            contras_valid = False
        else:
            contras_valid = True
        self.log_dict['contras_loss'] = contras_loss.item() if contras_valid else 0.0
        self.log_dict['pos_dist'] = pos_dist
        self.log_dict['neg_dist'] = neg_dist
        if contras_valid:
            l_g_total += contras_loss * self.contras_weight

        l_g_total.backward()
        self.optimizer_g.step()

    def test(self):
        self.net_g.eval()
        self.net_rrdb.eval()
        self.net_extractor.eval()
        with torch.no_grad():
            self.sife, self.sisr_in = self.net_rrdb(self.img_in_lq)
            self.features = self.net_extractor(self.match_img_in, self.img_ref)
            self.pre_flow, self.pre_offset, self.pre_sim, self.img_ref_feat = self.net_map(
                self.features, self.img_ref)
            self.output, self.mid_ref_feat = self.net_g(self.sisr_in, self.sife, self.pre_offset[0], self.pre_flow[0],
                                                        None, self.img_ref_feat)
        self.net_g.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['img_in_lq'] = self.img_in_lq.detach().cpu()
        out_dict['rlt'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        if self.net_d:
            self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

    def dist_validation(self, dataloader, current_iter, tb_logger,
                        save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger,
                                    save_img)


    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        logger = get_root_logger()
        pbar = ProgressBar(len(dataloader))
        avg_psnr = 0.
        avg_psnr_y = 0.
        avg_ssim_y = 0.
        dataset_name = dataloader.dataset.opt['name']

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data, train=False)
            self.test()

            visuals = self.get_current_visuals()
            sr_img, gt_img = tensor2img([visuals['rlt'], visuals['gt']])

            if 'padding' in val_data.keys():
                padding = val_data['padding']
                original_size = val_data['original_size']
                if padding:
                    sr_img = sr_img[:original_size[0], :original_size[1]]

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f"{img_name}_{self.opt['name']}.png")
                    if self.opt['suffix']:
                        save_img_path = save_img_path.replace(
                            '.png', f'_{self.opt["suffix"]}.png')
                imwrite(sr_img, save_img_path)

            # tentative for out of GPU memory
            del self.img_in_lq
            del self.output
            del self.gt
            torch.cuda.empty_cache()

            # calculate PSNR
            psnr = calculate_psnr(
                sr_img, gt_img, crop_border=self.opt['crop_border'], test_y_channel=False)
            avg_psnr += psnr
            psnr_y = calculate_psnr(
                sr_img, gt_img, crop_border=self.opt['crop_border'], test_y_channel=True)
            avg_psnr_y += psnr_y
            ssim_y = calculate_ssim(
                sr_img, gt_img, crop_border=self.opt['crop_border'], test_y_channel=True)
            avg_ssim_y += ssim_y

            if not self.is_train:
                logger.info(f'# img {img_name} # PSNR: {psnr:.4e} '
                            f'# PSNR_Y: {psnr_y:.4e} # SSIM_Y: {ssim_y:.4e}.')

            pbar.update(f'Test {img_name}')

        avg_psnr = avg_psnr / (idx + 1)
        avg_psnr_y = avg_psnr_y / (idx + 1)
        avg_ssim_y = avg_ssim_y / (idx + 1)

        # log
        logger.info(f'# Validation {dataset_name} # Iter: {current_iter} # PSNR: {avg_psnr:.4e} '
                    f'# PSNR_Y: {avg_psnr_y:.4e} # SSIM_Y: {avg_ssim_y:.4e}.')
        if tb_logger:
            tb_logger.add_scalar('psnr', avg_psnr, current_iter)
            tb_logger.add_scalar('psnr_y', avg_psnr_y, current_iter)
            tb_logger.add_scalar('ssim_y', avg_ssim_y, current_iter)
