from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models.vgg as vgg
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
@ARCH_REGISTRY.register()
class RefMaskCalculate(nn.Module):
    def __init__(self, shreshold=0.3, fixk=1):
        super(RefMaskCalculate, self).__init__()
        self.shreshold = shreshold
        self.fixk = fixk

    def forward(self, co_visible, pre_corr):
        batch, topk, h_ref, w_ref = pre_corr['relu3_1'].size()
        _, _, h, w = co_visible.size()
        device = co_visible.device
        ref_mask = torch.zeros((batch, topk, h_ref, w_ref), dtype=torch.bool, device=device)

        for b in range(batch):

            co_visible_in = F.pad(co_visible[b, 0], (0, w_ref - w, 0, h_ref - h))
            co_visible_ref = F.pad(co_visible[b, 1], (0, w_ref - w, 0, h_ref - h))


            co_visible_in_mask = co_visible_in > self.shreshold
            co_visible_ref_mask = co_visible_ref > self.shreshold

            true_count_in = co_visible_in_mask.sum()
            true_count_ref = co_visible_ref_mask.sum().clamp(min=1, max=topk*true_count_in)

            co_visible_in_mask_repeated = co_visible_in_mask.unsqueeze(0).repeat(topk, 1, 1)
            sim_flat = pre_corr['relu3_1'][b].view(-1)
            visible_indices = co_visible_in_mask_repeated.view(-1).nonzero(as_tuple=True)[0]

            if visible_indices.size(0) > 0:
                sim_values_at_visible = sim_flat[visible_indices]
                top_sim_values, top_sim_indices = torch.topk(sim_values_at_visible, true_count_ref.item(), dim=0)
                ref_mask_flat = ref_mask[b].view(-1)
                ref_mask_flat[visible_indices[top_sim_indices]] = True
                ref_mask[b] = ref_mask_flat.view(topk, h_ref, w_ref)
        ref_mask[:, :self.fixk] = True
        ref_mask_relu3 = ref_mask
        ref_mask_relu1 = ref_mask_relu3.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)
        return ref_mask_relu1


