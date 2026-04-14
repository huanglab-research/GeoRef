import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, make_layer


@ARCH_REGISTRY.register()
class AdaptiveAggregationNet(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module. It is used in EDVRNet.
    Args:
        nf (int): Number of the channels of middle features.
            Default: 64.
        ref_nf (int): Number of the channels of middle features.
            Default: 256.
    """

    def __init__(self,
                 ngf=64,
                 ref_ngf=64,
                 n_blocks=16):
        super().__init__()

        # multi-ref attention (before fusion conv)
        self.patch_size = 3
        channels = ref_ngf
        self.conv_emb1 = nn.Sequential(
            nn.Conv2d(ngf, channels, 1),
            nn.PReLU())
        self.conv_emb2 = nn.Sequential(
            nn.Conv2d(ref_ngf, channels,
                      self.patch_size, 1, self.patch_size // 2),
            nn.PReLU())
        self.conv_ass = nn.Conv2d(ref_ngf, ref_ngf,
                                  self.patch_size, 1, self.patch_size // 2)
        self.scale = channels ** -0.5
        self.feat_fusion = nn.Conv2d(
            ngf + channels * 2, ngf, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.unet_head = nn.Conv2d(3, ngf, kernel_size=3, stride=1, padding=1)
        self.body = make_layer(ResidualBlockNoBN, n_blocks, num_feat=ngf)
        self.tail = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1))

    def spatial_padding(self, feats):
        _, _, h, w = feats.size()
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        feats = F.pad(feats, [0, pad_w, 0, pad_h], mode='reflect')
        return feats


    def forward(self, target, refs, refs_mask=None):
        in_feat = self.unet_head(target)
        n, _, h_input, w_input = in_feat.size()
        n, t, _, h_input, w_input = refs.size()

        # check_mask = refs_mask.sum(dim=1).detach().cpu().numpy()
        # in_feat = self.spatial_padding(in_feat)
        refs = refs.flatten(0, 1)
        # refs = self.spatial_padding(refs)
        # multi-ref attention
        embedding_in = self.conv_emb1(in_feat) * self.scale  # (n, c, h, w)
        embedding_in = embedding_in.permute(0, 2, 3, 1).unsqueeze(3)  # (n, h, w, 1, c)
        embedding_in = embedding_in.contiguous().flatten(0, 2)  # (n*h*w, 1, c)
        emb = self.conv_emb2(refs).unflatten(0, (n, t))  # (n, t, c, h, w)
        emb = emb.permute(0, 3, 4, 2, 1)  # (n, h, w, c, t)
        emb = emb.contiguous().flatten(0, 2)  # (n*h*w, c, t)
        ass = self.conv_ass(refs).unflatten(0, (n, t))  # (n, t, c*2, h, w)
        ass = ass.permute(0, 3, 4, 1, 2)  # (n, h, w, t, c*2)
        ass = ass.contiguous().flatten(0, 2)  # (n*h*w, t, c*2)

        corr_prob = torch.matmul(embedding_in, emb)  # (n*h*w, 1, t)
        if refs_mask is not None:
            refs_mask = rearrange(refs_mask, "n t h w -> (n h w) 1 t")
            # refs_mask = refs_mask.unsqueeze(2).expand(-1, -1, self.nhead, -1, -1)
            mask = torch.where(refs_mask, torch.tensor(0.0, device=refs_mask.device),
                               torch.tensor(-100.0, device=refs_mask.device))

            corr_prob = corr_prob + mask
        else:
            corr_prob = corr_prob
        corr_prob = F.softmax(corr_prob, dim=2)
        refs = torch.matmul(corr_prob, ass).squeeze(1)  # (n*h*w, c*2)
        refs = refs.unflatten(0, (n, *in_feat.shape[-2:]))  # (n, h, w, c*2)
        refs = refs.permute(0, 3, 1, 2).contiguous()  # (n, c*2, h, w)
        refs = refs[:, :, :h_input, :w_input]
        refs = self.body(refs)
        refs = self.tail(refs)
        return refs + target