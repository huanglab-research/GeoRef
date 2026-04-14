import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_

from einops import rearrange
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, default_init_weights, make_layer, srntt_init_weights, ResidualDDFBlock


class DynAgg(ModulatedDeformConv2d):
    '''
    Use other features to generate offsets and masks.
    Intialized the offset with precomputed non-local offset.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 deform_groups=1,
                 extra_offset_mask=True,
                 max_residue_magnitude=10,
                 use_sim=False
                 ):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, deform_groups)
        self.extra_offset_mask = extra_offset_mask
        self.max_residue_magnitude = max_residue_magnitude
        self.use_sim = use_sim

        channels_ = self.deform_groups * 3 * self.kernel_size[
            0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x, pre_offset, pre_sim=None):
        '''
        Args:
            pre_offset: precomputed_offset. Size: [b, 9, h, w, 2]
        '''
        if self.extra_offset_mask:
            # x = [input, features]
            out = self.conv_offset_mask(x[1])
            x = x[0]
        else:
            out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        # if self.max_residue_magnitude:
        #     offset = self.max_residue_magnitude * torch.tanh(offset)
        # repeat pre_offset along dim1, shape: [b, 9*groups, h, w, 2]
        pre_offset = pre_offset.repeat([1, self.deform_groups, 1, 1, 1])
        # the order of offset is [y, x, y, x, ..., y, x]
        pre_offset_reorder = torch.zeros_like(offset)
        # add pre_offset on y-axis
        pre_offset_reorder[:, 0::2, :, :] = pre_offset[:, :, :, :, 1]
        # add pre_offset on x-axis
        pre_offset_reorder[:, 1::2, :, :] = pre_offset[:, :, :, :, 0]
        offset = offset + pre_offset_reorder
        # print(offset.size())
        if pre_sim is not None:
            mask = torch.sigmoid(mask*pre_sim)
        else:
            mask = torch.sigmoid(mask)  # [9, 72, 40, 40]

        offset_mean = torch.mean(torch.abs(offset - pre_offset_reorder))
        if offset_mean > 100:
            logger.warning(
                'Offset mean is {}, larger than 100.'.format(offset_mean))
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                           self.stride, self.padding, self.dilation,
                           self.groups, self.deform_groups)


@ARCH_REGISTRY.register()
class ResblockAlignmentNet(nn.Module):
    def __init__(self,
                 ngf=64,
                 groups=8,
                 n_blocks=16,
                 ):
        super(ResblockAlignmentNet, self).__init__()
        self.dyn_agg_restore = DynamicAggregationRestoration( ngf=ngf, groups=groups, n_blocks=n_blocks)
        srntt_init_weights(self, init_type='normal', init_gain=0.02)
        self.re_init_dcn_offset()

    def re_init_dcn_offset(self):
        self.dyn_agg_restore.down_medium_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.down_medium_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.down_large_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.down_large_dyn_agg.conv_offset_mask.bias.data.zero_()

        self.dyn_agg_restore.up_small_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.up_small_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.up_medium_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.up_medium_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.up_large_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.up_large_dyn_agg.conv_offset_mask.bias.data.zero_()

    def forward(self, x, sife, pre_offset, pre_flow, pre_sim, img_ref_feat):
        """
        Args:
            x (Tensor): the input image of SRNTT.
            maps (dict[Tensor]): the swapped feature maps on relu3_1, relu2_1
                and relu1_1. depths of the maps are 256, 128 and 64
                respectively.
        """
        base = F.interpolate(x, None, 4, 'bilinear', False)
        ref_align, middle_feat = self.dyn_agg_restore(base, sife, pre_offset, pre_flow, pre_sim, img_ref_feat)

        return ref_align+base, middle_feat

class DynamicAggregationRestoration(nn.Module):

        def __init__(self,
                     ngf=64,
                     groups=8,
                     n_blocks=16,
                     ):
            super(DynamicAggregationRestoration, self).__init__()

            self.unet_head = nn.Conv2d(3, ngf, kernel_size=3, stride=1, padding=1)

            # ---------------------- Down ----------------------
            # dynamic aggregation module for relu1_1 reference feature
            self.down_large_offset_conv1 = nn.Conv2d( ngf + 64*2, 64, 3, 1, 1, bias=True)
            self.down_large_offset_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
            self.down_large_dyn_agg = DynAgg(64, 64, 3, stride=1, padding=1, dilation=1,
                                             deform_groups=groups, extra_offset_mask=True)

            # for large scale
            self.down_head_large = nn.Sequential(
                nn.Conv2d(ngf + 64, ngf, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.1, True))
            self.ddf_block_1 = make_layer(ResidualDDFBlock, 2, nf=ngf)
            self.down_body_large = make_layer(ResidualBlockNoBN, n_blocks, num_feat=ngf)
            self.down_tail_large = nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1)

            # dynamic aggregation module for relu2_1 reference feature
            self.down_medium_offset_conv1 = nn.Conv2d(
                ngf + 128*2, 128, 3, 1, 1, bias=True)
            self.down_medium_offset_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
            self.down_medium_dyn_agg = DynAgg(128, 128, 3, stride=1, padding=1, dilation=1,
                                              deform_groups=groups, extra_offset_mask=True)
            # for medium scale restoration
            self.down_head_medium = nn.Sequential(
                nn.Conv2d(ngf + 128, ngf, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.1, True))
            self.ddf_block_2 = make_layer(ResidualDDFBlock, 2, nf=ngf)
            self.down_body_medium = make_layer(ResidualBlockNoBN, n_blocks, num_feat=ngf)

            self.down_tail_medium = nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1)

            # ---------------------- Up ----------------------
            # dynamic aggregation module for relu3_1 reference feature
            self.up_small_offset_conv1 = nn.Conv2d(
                ngf + 256*2, 256, 3, 1, 1, bias=True)  # concat for diff
            self.up_small_offset_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=True)
            self.up_small_dyn_agg = DynAgg(256, 256, 3, stride=1, padding=1, dilation=1,
                                           deform_groups=groups, extra_offset_mask=True)
            # for small scale restoration
            self.up_head_small = nn.Sequential(
                nn.Conv2d(ngf + 256, ngf, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.1, True))
            self.ddf_block_3 = make_layer(ResidualDDFBlock, 2, nf=ngf)
            self.up_body_small = make_layer(ResidualBlockNoBN, n_blocks, num_feat=ngf)

            self.up_tail_small = nn.Sequential(
                nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))

            # dynamic aggregation module for relu2_1 reference feature
            self.up_medium_offset_conv1 = nn.Conv2d(
                ngf + 128*2, 128, 3, 1, 1, bias=True)
            self.up_medium_offset_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
            self.up_medium_dyn_agg = DynAgg(128, 128, 3, stride=1, padding=1, dilation=1,
                                            deform_groups=groups, extra_offset_mask=True)
            # for medium scale restoration
            self.up_head_medium = nn.Sequential(
                nn.Conv2d(ngf + 128, ngf, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.1, True))
            self.ddf_block_4 = make_layer(ResidualDDFBlock, 2, nf=ngf)
            self.up_body_medium = make_layer(ResidualBlockNoBN, n_blocks, num_feat=ngf)

            self.up_tail_medium = nn.Sequential(
                nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))

            # dynamic aggregation module for relu1_1 reference feature
            self.up_large_offset_conv1 = nn.Conv2d(ngf + 64*2, 64, 3, 1, 1, bias=True)
            self.up_large_offset_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
            self.up_large_dyn_agg = DynAgg(64, 64, 3, stride=1, padding=1, dilation=1,
                                           deform_groups=groups, extra_offset_mask=True)

            # for large scale
            self.up_head_large = nn.Sequential(
                nn.Conv2d(ngf + 64, ngf, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.1, True))
            self.ddf_block_5 = make_layer(ResidualDDFBlock, 2, nf=ngf)
            self.up_body_large = make_layer(ResidualBlockNoBN, n_blocks, num_feat=ngf)
            self.up_tail_large = nn.Sequential(
                nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(ngf // 2, 3, kernel_size=3, stride=1, padding=1))
            self.high_q_conv1 = nn.Conv2d(ngf+64, ngf, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        def flow_warp(self,
                      x,
                      flow,
                      interp_mode='bilinear',
                      padding_mode='zeros',
                      align_corners=True):
            """Warp an image or feature map with optical flow.
            Args:
                x (Tensor): Tensor with size (n, c, h, w).
                flow (Tensor): Tensor with size (n, h, w, 2), normal value.
                interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
                padding_mode (str): 'zeros' or 'border' or 'reflection'.
                    Default: 'zeros'.
                align_corners (bool): Before pytorch 1.3, the default value is
                    align_corners=True. After pytorch 1.3, the default value is
                    align_corners=False. Here, we use the True as default.
            Returns:
                Tensor: Warped image or feature map.
            """

            assert x.size()[-2:] == flow.size()[1:3]
            _, _, h, w = x.size()
            # create mesh grid
            grid_y, grid_x = torch.meshgrid(
                torch.arange(0, h).type_as(x),
                torch.arange(0, w).type_as(x))
            grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
            grid.requires_grad = False

            vgrid = grid + flow
            # scale grid to [-1,1]
            vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
            vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
            vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
            output = F.grid_sample(x,
                                   vgrid_scaled,
                                   mode=interp_mode,
                                   padding_mode=padding_mode,
                                   align_corners=align_corners)

            return output

        def forward(self, base, sife, pre_offset, pre_flow, pre_sim, img_ref_feat):
            # dynamic aggregation for relu3_1 reference feature
            pre_relu1_swapped_feat = self.flow_warp(img_ref_feat['relu1_1'], pre_flow['relu1_1'])
            pre_relu2_swapped_feat = self.flow_warp(img_ref_feat['relu2_1'], pre_flow['relu2_1'])
            pre_relu3_swapped_feat = self.flow_warp(img_ref_feat['relu3_1'], pre_flow['relu3_1'])
            x0 = self.unet_head(base)  # [B, 64, 160, 160]
            # -------------- Down ------------------
            # large scale
            down_relu1_offset = torch.cat([x0, pre_relu1_swapped_feat, img_ref_feat['relu1_1']], 1)
            down_relu1_offset = self.lrelu(self.down_large_offset_conv1(down_relu1_offset))
            down_relu1_offset = self.lrelu(self.down_large_offset_conv2(down_relu1_offset))
            down_relu1_swapped_feat = self.lrelu(
                self.down_large_dyn_agg([img_ref_feat['relu1_1'], down_relu1_offset],
                                        pre_offset['relu1_1']))

            h = torch.cat([x0, down_relu1_swapped_feat], 1)
            h = self.down_head_large(h)
            h = self.ddf_block_1(h) + x0
            h = self.down_body_large(h) + x0
            x1 = self.down_tail_large(h)  # [B, 64, 80, 80]

            # medium scale
            down_relu2_offset = torch.cat([x1, pre_relu2_swapped_feat, img_ref_feat['relu2_1']], 1)
            down_relu2_offset = self.lrelu(self.down_medium_offset_conv1(down_relu2_offset))
            down_relu2_offset = self.lrelu(self.down_medium_offset_conv2(down_relu2_offset))
            down_relu2_swapped_feat = self.lrelu(
                self.down_medium_dyn_agg([img_ref_feat['relu2_1'], down_relu2_offset],
                                         pre_offset['relu2_1']))

            h = torch.cat([x1, down_relu2_swapped_feat], 1)
            h = self.down_head_medium(h)
            h = self.ddf_block_2(h) + x1
            h = self.down_body_medium(h) + x1
            x2 = self.down_tail_medium(h)  # [9, 64, 40, 40]

            # -------------- Up ------------------

            # dynamic aggregation for relu3_1 reference feature
            relu3_offset = torch.cat([x2, pre_relu3_swapped_feat, img_ref_feat['relu3_1']], 1)
            relu3_offset = self.lrelu(self.up_small_offset_conv1(relu3_offset))
            relu3_offset = self.lrelu(self.up_small_offset_conv2(relu3_offset))
            relu3_swapped_feat = self.lrelu(
                self.up_small_dyn_agg([img_ref_feat['relu3_1'], relu3_offset], pre_offset['relu3_1']))

            x2_agg = torch.cat([sife, x2], dim=1)
            x2_agg = self.lrelu(self.high_q_conv1(x2_agg))

            # small scale
            h = torch.cat([x2_agg, relu3_swapped_feat], 1)
            h = self.up_head_small(h)
            h = self.ddf_block_3(h) + x2
            h = self.up_body_small(h) + x2
            x = self.up_tail_small(h)  # [9, 64, 80, 80]

            # dynamic aggregation for relu2_1 reference feature
            relu2_offset = torch.cat([x, pre_relu2_swapped_feat, img_ref_feat['relu2_1']], 1)
            relu2_offset = self.lrelu(self.up_medium_offset_conv1(relu2_offset))
            relu2_offset = self.lrelu(self.up_medium_offset_conv2(relu2_offset))
            relu2_swapped_feat = self.lrelu(
                self.up_medium_dyn_agg([img_ref_feat['relu2_1'], relu2_offset],
                                       pre_offset['relu2_1']))

            # medium scale
            h = torch.cat([x + x1, relu2_swapped_feat], 1)
            h = self.up_head_medium(h)
            h = self.ddf_block_4(h) + x
            h = self.up_body_medium(h) + x
            x = self.up_tail_medium(h)  # [9, 64, 160, 160]

            # dynamic aggregation for relu1_1 reference feature
            relu1_offset = torch.cat([x, pre_relu1_swapped_feat, img_ref_feat['relu1_1']], 1)
            relu1_offset = self.lrelu(self.up_large_offset_conv1(relu1_offset))
            relu1_offset = self.lrelu(self.up_large_offset_conv2(relu1_offset))
            relu1_swapped_feat = self.lrelu(
                self.up_large_dyn_agg([img_ref_feat['relu1_1'], relu1_offset],
                                      pre_offset['relu1_1']))

            # large scale
            h = torch.cat([x + x0, relu1_swapped_feat], 1)
            h = self.up_head_large(h)
            h = self.ddf_block_5(h) + x
            h = self.up_body_large(h) + x
            out = self.up_tail_large(h)

            return out, h