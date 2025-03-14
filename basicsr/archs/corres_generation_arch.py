import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

from .arch_util import tensor_shift
from .ref_map_util import feature_match_index, feature_match_topk_index
from .vgg_arch import VGGFeatureExtractor


@ARCH_REGISTRY.register()
class CorrespondenceGenerationArch(nn.Module):

    def __init__(self,
                 patch_size=3,
                 stride=1,
                 topk=3,
                 vgg_layer_list=['relu3_1', 'relu2_1', 'relu1_1'],
                 vgg_type='vgg19'):
        super(CorrespondenceGenerationArch, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.padding = patch_size // 2
        self.topk = topk
        self.vgg_layer_list = vgg_layer_list
        self.vgg = VGGFeatureExtractor(
            layer_name_list=vgg_layer_list, vgg_type=vgg_type)

    def index_to_flow(self, max_idx):
        device = max_idx.device
        # max_idx to flow
        h, w = max_idx.size()
        flow_w = max_idx % w
        flow_h = max_idx // w

        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h).to(device),
            torch.arange(0, w).to(device))
        grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).float().to(device)
        grid.requires_grad = False
        flow = torch.stack((flow_w, flow_h),
                           dim=2).unsqueeze(0).float().to(device)
        flow = flow - grid  # shape:(1, w, h, 2)
        flow = torch.nn.functional.pad(flow, (0, 0, 0, 2, 0, 2))

        return flow

    def forward(self, dense_features, img_ref_hr):
        batch_flow_relu3 = []
        batch_flow_relu2 = []
        batch_flow_relu1 = []
        batch_offset_relu3 = []
        batch_offset_relu2 = []
        batch_offset_relu1 = []
        batch_similarity_relu3 = []
        batch_similarity_relu2 = []
        batch_similarity_relu1 = []

        for ind in range(img_ref_hr.size(0)):
            feat_in = dense_features['dense_features1'][ind]
            feat_ref = dense_features['dense_features2'][ind]
            c, h, w = feat_in.size()
            feat_in = F.normalize(feat_in.reshape(c, -1), dim=0).view(c, h, w)
            feat_ref = F.normalize(
                feat_ref.reshape(c, -1), dim=0).view(c, h, w)

            top_idx, top_val = feature_match_topk_index(
                feat_in,
                feat_ref,
                topk=self.topk,
                patch_size=self.patch_size,
                padding = 0,
                input_stride=self.stride,
                ref_stride=self.stride,
                is_norm=True,
                norm_input=True)

            k_offset_relu3, k_flows_relu3, k_similarity_relu3 = [], [], []
            for k in range(top_idx.size(0)):
                sim = F.pad(top_val[k], (1, 1, 1, 1)).unsqueeze(0)
                # sim = top_val[k].unsqueeze(0)
                flow = self.index_to_flow(top_idx[k])
                shifted_offset = []
                for i in range(0, 3):
                    for j in range(0, 3):
                        flow_shift = tensor_shift(flow, (i, j))
                        shifted_offset.append(flow_shift)
                shifted_offset = torch.cat(shifted_offset, dim=0)
                k_flows_relu3.append(flow)
                k_offset_relu3.append(shifted_offset)
                k_similarity_relu3.append(sim)
            k_flows_relu3 = torch.cat(k_flows_relu3, dim=0)
            k_offset_relu3 = torch.stack(k_offset_relu3, dim=0)
            k_similarity_relu3 = torch.cat(k_similarity_relu3, dim=0)
            batch_flow_relu3.append(k_flows_relu3)  # shape:(k, w, h, 2)
            batch_offset_relu3.append(k_offset_relu3)  # shape:(k, 9, w, h, 2)
            batch_similarity_relu3.append(k_similarity_relu3)  # shape:(k, w, h)

            # offset map for relu2_1
            k_flows_relu2 = torch.repeat_interleave(k_flows_relu3, 2, 1)
            k_flows_relu2 = torch.repeat_interleave(k_flows_relu2, 2, 2)
            k_flows_relu2 *= 2
            k_similarity_relu2 = torch.repeat_interleave(k_similarity_relu3, 2, 1)
            k_similarity_relu2 = torch.repeat_interleave(k_similarity_relu2, 2, 2)
            # shift offset relu2
            k_offset_relu2 = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift = tensor_shift(k_flows_relu2, (i * 2, j * 2))
                    k_offset_relu2.append(flow_shift)
            k_offset_relu2 = torch.stack(k_offset_relu2, dim=1)
            batch_flow_relu2.append(k_flows_relu2)
            batch_offset_relu2.append(k_offset_relu2)  # shape:(k, 9, w, h, 2)
            batch_similarity_relu2.append(k_similarity_relu2)  # shape:(k, w, h)

            # offset map for relu1_1
            k_flows_relu1 = torch.repeat_interleave(k_flows_relu3, 4, 1)
            k_flows_relu1 = torch.repeat_interleave(k_flows_relu1, 4, 2)
            k_flows_relu1 *= 4
            k_similarity_relu1 = torch.repeat_interleave(k_similarity_relu3, 4, 1)
            k_similarity_relu1 = torch.repeat_interleave(k_similarity_relu1, 4, 2)
            # shift offset relu1
            k_offset_relu1 = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift = tensor_shift(k_flows_relu1, (i * 4, j * 4))
                    k_offset_relu1.append(flow_shift)
            k_offset_relu1 = torch.stack(k_offset_relu1, dim=1)
            batch_flow_relu1.append(k_flows_relu1)
            batch_offset_relu1.append(k_offset_relu1)  # shape:(k, 9, w, h, 2)
            batch_similarity_relu1.append(k_similarity_relu1)  # shape:(k, w, h)

        batch_flow_relu3 = torch.stack(batch_flow_relu3, dim=0)
        batch_flow_relu2 = torch.stack(batch_flow_relu2, dim=0)
        batch_flow_relu1 = torch.stack(batch_flow_relu1, dim=0)
        batch_offset_relu3 = torch.stack(batch_offset_relu3, dim=0)
        batch_offset_relu2 = torch.stack(batch_offset_relu2, dim=0)
        batch_offset_relu1 = torch.stack(batch_offset_relu1, dim=0)
        batch_similarity_relu3 = torch.stack(batch_similarity_relu3, dim=0)
        batch_similarity_relu2 = torch.stack(batch_similarity_relu2, dim=0)
        batch_similarity_relu1 = torch.stack(batch_similarity_relu1, dim=0)


        pre_flow = [
            {
                'relu1_1': batch_flow_relu1[:, k],
                'relu2_1': batch_flow_relu2[:, k],
                'relu3_1': batch_flow_relu3[:, k]
            }
            for k in range(self.topk)
        ]

        pre_offset = [
            {
                'relu1_1': batch_offset_relu1[:, k],
                'relu2_1': batch_offset_relu2[:, k],
                'relu3_1': batch_offset_relu3[:, k]
            }
            for k in range(self.topk)
        ]
        pre_similarity = {}
        pre_similarity['relu1_1'] = batch_similarity_relu1
        pre_similarity['relu2_1'] = batch_similarity_relu2
        pre_similarity['relu3_1'] = batch_similarity_relu3

        img_ref_feat = self.vgg(img_ref_hr)
        return pre_flow, pre_offset, pre_similarity, img_ref_feat
