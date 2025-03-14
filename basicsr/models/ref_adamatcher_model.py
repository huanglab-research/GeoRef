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
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel

from basicsr.utils.registry import MODEL_REGISTRY

loss_module = importlib.import_module('basicsr.models.losses')


@MODEL_REGISTRY.register()
class MultiAdamatcherModel(BaseModel):

    def __init__(self, opt):
        super(MultiAdamatcherModel, self).__init__(opt)
        # define network for adamatcher
        self.net_adamatcher = build_network(opt['network_adamatcher'])
        self.net_adamatcher = self.model_to_device(self.net_adamatcher)
        # self.print_network(self.net_adamatcher)
        # load pretrained adamatcher
        load_path = self.opt['path'].get('pretrain_network_adamatcher', None)
        if load_path is not None:
            self.load_adamatcher_network(self.net_adamatcher, load_path,
                                   self.opt['path']['strict_load'])
            for param in self.net_adamatcher.parameters():
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

        # check the schedulers
        self.setup_schedulers()

        self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.img_in_lq = data['img_in_lq'].to(self.device)
        self.img_ref_list = data['img_ref_list'].to(self.device)
        self.img_ref_list = list(torch.unbind(self.img_ref_list, dim=1))
        self.gt = data['img_in'].to(self.device)  # gt
        self.match_img_in = data['img_in_up'].to(self.device)
        self.img_in_mask = data['img_in_mask'].to(self.device)
        self.img_ref_mask_list = data['img_ref_mask_list'].to(self.device)
        # self.co_visible = data['co_visible'].to(self.device)

    def optimize_parameters(self, step):
        self.features = self.net_extractor(self.match_img_in, self.img_ref_list)
        self.img_ref_feat = self.net_vgg(self.img_ref_list)
        self.base_feat = self.net_rrdb(self.img_in_lq)
        # self.co_visible = self.net_adamatcher(self.match_img_in, self.img_ref_list)
        self.pre_corr = self.net_map(self.features)
        self.corr_mask = self.net_mask(self.co_visible, self.pre_corr)
        self.output = self.net_g(self.img_in_lq, self.base_feat, self.pre_corr, self.corr_mask, self.img_ref_feat)


        if step <= self.net_g_pretrain_steps:
            # pretrain the net_g with pixel Loss
            self.optimizer_g.zero_grad()
            l_pix = self.cri_pix(self.output, self.gt)
            l_pix.backward()
            self.optimizer_g.step()

            # set log
            self.log_dict['l_pix'] = l_pix.item()
        else:
            if self.net_d:
                # train net_d
                self.optimizer_d.zero_grad()
                for p in self.net_d.parameters():
                    p.requires_grad = True

                # compute WGAN loss
                real_d_pred = self.net_d(self.gt)
                l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                self.log_dict['l_d_real'] = l_d_real.item()
                self.log_dict['out_d_real'] = torch.mean(real_d_pred.detach())
                # fake
                fake_d_pred = self.net_d(self.output.detach())
                l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                self.log_dict['l_d_fake'] = l_d_fake.item()
                self.log_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
                l_d_total = l_d_real + l_d_fake
                if self.cri_grad_penalty:
                    l_grad_penalty = self.cri_grad_penalty(
                        self.net_d, self.gt, self.output)
                    self.log_dict['l_grad_penalty'] = l_grad_penalty.item()
                    l_d_total += l_grad_penalty
                l_d_total.backward()
                self.optimizer_d.step()

            # train net_g
            self.optimizer_g.zero_grad()
            if self.net_d:
                for p in self.net_d.parameters():
                    p.requires_grad = False

            l_g_total = 0
            if (step - self.net_g_pretrain_steps) % self.net_d_steps == 0 and (
                    step - self.net_g_pretrain_steps) > self.net_d_init_steps:
                if self.cri_pix:
                    l_g_pix = self.cri_pix(self.output, self.gt)
                    l_g_total += l_g_pix
                    self.log_dict['l_g_pix'] = l_g_pix.item()
                if self.cri_perceptual:
                    l_g_percep, _ = self.cri_perceptual(self.output, self.gt)
                    l_g_total += l_g_percep
                    self.log_dict['l_g_percep'] = l_g_percep.item()
                if self.cri_style:
                    _, l_g_style = self.cri_style(self.output, self.gt)
                    l_g_total += l_g_style
                    self.log_dict['l_g_style'] = l_g_style.item()
                if self.cri_texture:
                    l_g_texture = self.cri_texture(self.output, self.maps,
                                                   self.weights)
                    l_g_total += l_g_texture
                    self.log_dict['l_g_texture'] = l_g_texture.item()

                if self.net_d:
                    # gan loss
                    fake_g_pred = self.net_d(self.output)
                    l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                    l_g_total += l_g_gan
                    self.log_dict['l_g_gan'] = l_g_gan.item()

                l_g_total.backward()
                self.optimizer_g.step()

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.features = self.net_extractor(self.match_img_in, self.img_ref_list)
            self.img_ref_feat = self.net_vgg(self.img_ref_list)
            self.base_feat = self.net_rrdb(self.img_in_lq)
            self.co_visible = self.net_adamatcher(self.match_img_in, self.img_ref_list, self.img_in_mask, self.img_ref_mask_list)
            self.pre_offset = self.net_map(self.features)
            self.output = self.net_g(self.img_in_lq, self.base_feat, self.pre_offset, self.img_ref_feat)
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

        start_time = time.time()
        path = '/home/huangl/zk/Code_PycharmSSH/datasets/CUFED/test/CUFED_Covisible'

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()

            # series = val_data['in_path'][0].split('/')[-2]
            # save_path = osp.join(path, series)
            save_path = path
            # 提前创建好目录
            if not osp.exists(save_path):
                os.makedirs(save_path, exist_ok=True)

            # 并行保存
            with ThreadPoolExecutor() as executor:
                futures = []
                for i in range(len(val_data['ref_paths'])):
                    np_array = self.co_visible[i].cpu().numpy()
                    inname, refname = self.get_inname_refname(val_data, i)
                    save_img_path = osp.join(save_path, f'{inname}_with_{refname}.npy')

                    # 提交保存任务给线程池
                    futures.append(executor.submit(self.save_npy, np_array, save_img_path))

                # 等待所有任务完成
                for future in futures:
                    future.result()

            # 计算剩余时间
            total_time = time.time() - start_time
            time_sec_avg = total_time / (idx + 1)
            eta_sec = time_sec_avg * (len(dataloader) - idx - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            print(f'Estimated remaining time: {eta_str}')

            # # tentative for out of GPU memory
            # del self.img_in_lq
            # del self.output
            # del self.gt
            # torch.cuda.empty_cache()


    def get_inname_refname(self, val_data, i):
        inname = val_data['in_path'][0].split('/')[-1].split('.')[0]
        refname = val_data['ref_paths'][i][0].split('/')[-1].split('.')[0]
        # p_in = val_data['in_p']  # 定义你的 p_in
        # p_ref = val_data['ref_ps'][i]  # 定义你的 p_ref
        # p_in_str = f'{p_in[0][0]}_{p_in[0][1]}'
        # p_ref_str = f'{p_ref[0][0]}_{p_ref[0][1]}'
        # inname = f'{inname}_p({p_in_str})'
        # refname = f'{refname}_p({p_ref_str})'
        return inname, refname

    def save_npy(self, np_array, save_img_path):
        np.save(save_img_path, np_array)