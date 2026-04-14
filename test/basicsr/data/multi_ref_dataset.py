from ast import literal_eval
import cv2
import glob
import mmcv
import numpy as np
import os.path as osp
import pandas as pd
import random
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

from .transforms import augment, mod_crop, totensor
from basicsr.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class MultiRefCUFEDSet(data.Dataset):
    def __init__(self, opt):
        super(MultiRefCUFEDSet, self).__init__()
        self.opt = opt

        self.in_folder, self.ref_folder = opt['dataroot_in'], opt[
            'dataroot_ref']
        self.ann_file = opt['ann_file']
        self.load_annotations()
    def get_inname_refname(self, input, ref):
        inname_base = input.split('/')[-1].split('.')[0]
        refname_base = ref.split('/')[-1].split('.')[0]
        return f'{inname_base}_with_{refname_base}.npy'
    def load_annotations(self):
        self.samples = []
        df = pd.read_csv(self.ann_file)
        for i in range(len(df)):
            target, H, M1, M2, L1, L2 = df.loc[i].tolist()
            target = osp.join(self.in_folder,  target)
            references = [
                osp.join(self.in_folder, H),
                osp.join(self.in_folder, M1),
                osp.join(self.in_folder, M2),
                osp.join(self.in_folder, L1),
                osp.join(self.in_folder, L2)]

            self.samples.append((target, references))
        print(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scale = self.opt['scale']
        in_path, ref_paths= self.samples[idx]

        img_in = cv2.imread(in_path)
        Refs_origin = [cv2.imread(ref_path) for ref_path in ref_paths]

        img_in = mod_crop(img_in, scale)
        img_in_gt = img_in.copy()
        img_in_h, img_in_w, _ = img_in.shape

        Refs_origin = [mod_crop(ref, scale) for ref in Refs_origin]
        img_ref_h = max(ref.shape[0] for ref in Refs_origin)
        img_ref_w = max(ref.shape[1] for ref in Refs_origin)

        gt_h = max(img_in_h, img_ref_h)
        gt_w = max(img_in_w, img_ref_w)

        padding = True
        img_in = mmcv.impad(img_in, shape=(gt_h, gt_w), pad_val=0)
        Refs_origin = [
            mmcv.impad(img_ref, shape=(gt_h, gt_w), pad_val=0)
            for img_ref in Refs_origin
        ]
        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale
        img_in_lq = Image.fromarray(img_in).resize((lq_w, lq_h), Image.BICUBIC)
        img_in_up = img_in_lq.resize((gt_w, gt_h), Image.BICUBIC)
        Refs_lq = []
        Refs_up = []
        for img_ref in Refs_origin:
            img_ref_lq = Image.fromarray(img_ref).resize((lq_w, lq_h), Image.BICUBIC)
            img_ref_up = img_ref_lq.resize((gt_w, gt_h), Image.BICUBIC)
            Refs_lq.append(img_ref_lq)
            Refs_up.append(img_ref_up)

        img_in = img_in.astype(np.float32) / 255.
        img_in_gt = img_in_gt.astype(np.float32) / 255.
        img_in_lq = np.array(img_in_lq).astype(np.float32) / 255.
        img_in_up = np.array(img_in_up).astype(np.float32) / 255.
        Refs = [img_ref.astype(np.float32) / 255. for img_ref in Refs_origin]
        Refs_lq = [np.array(img_ref_lq).astype(np.float32) / 255. for img_ref_lq in Refs_lq]
        Refs_up = [np.array(img_ref_up).astype(np.float32) / 255. for img_ref_up in Refs_up]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_in_lq, img_in_up, img_in_gt = img2tensor(  # noqa: E501
            [img_in, img_in_lq, img_in_up, img_in_gt],
            bgr2rgb=True,
            float32=True)
        Refs = img2tensor(Refs, bgr2rgb=True, float32=True)
        Refs_lq = img2tensor(Refs_lq, bgr2rgb=True, float32=True)
        Refs_up = img2tensor(Refs_up, bgr2rgb=True, float32=True)
        Refs = torch.stack(Refs)
        Refs_lq = torch.stack(Refs_lq)
        Refs_up = torch.stack(Refs_up)

        return_dict = {
            'img_in': img_in_gt,
            'img_in_lq': img_in_lq,
            'img_in_up': img_in_up,
            'img_ref_list': Refs,
            'img_ref_lq_list': Refs_lq,
            'img_ref_up_list': Refs_up,
            'lq_path': in_path,
            'padding': padding,
            'original_size': (img_in_h, img_in_w),
        }

        return return_dict


def image_pair_generation_perspective(img,
                          random_perturb_range=(0, 32),
                          cropping_window_size=160,
                          dsize=None):

    if img is not None:
        shape1 = img.shape
        h = shape1[0]
        w = shape1[1]
    else:
        h = 160
        w = 160

    # ===== in image-1
    cropS = cropping_window_size
    x_topleft = np.random.randint(random_perturb_range[1],
                                  max(w, w - cropS - random_perturb_range[1]))
    y_topleft = np.random.randint(random_perturb_range[1],
                                  max(h, h - cropS - random_perturb_range[1]))

    x_topright = x_topleft + cropS
    y_topright = y_topleft

    x_bottomleft = x_topleft
    y_bottomleft = y_topleft + cropS

    x_bottomright = x_topleft + cropS
    y_bottomright = y_topleft + cropS

    tl = (x_topleft, y_topleft)
    tr = (x_topright, y_topright)
    br = (x_bottomright, y_bottomright)
    bl = (x_bottomleft, y_bottomleft)

    rect1 = np.array([tl, tr, br, bl], dtype=np.float32)

    # ===== in image-2
    x2_topleft = x_topleft + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])
    y2_topleft = y_topleft + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])

    x2_topright = x_topright + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])
    y2_topright = y_topright + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])

    x2_bottomleft = x_bottomleft + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])
    y2_bottomleft = y_bottomleft + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])

    x2_bottomright = x_bottomright + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])
    y2_bottomright = y_bottomright + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])

    tl2 = (x2_topleft, y2_topleft)
    tr2 = (x2_topright, y2_topright)
    br2 = (x2_bottomright, y2_bottomright)
    bl2 = (x2_bottomleft, y2_bottomleft)

    rect2 = np.array([tl2, tr2, br2, bl2], dtype=np.float32)

    # ===== homography
    H = cv2.getPerspectiveTransform(src=rect1, dst=rect2)
    H_inverse = np.linalg.inv(H)

    if img is not None:
        if dsize is None:
            dsize = (w, h)
        img_warped = cv2.warpPerspective(src=img, M=H_inverse, dsize=dsize, flags=cv2.INTER_CUBIC)
        return img_warped, H, H_inverse
    else:
        return H_inverse
