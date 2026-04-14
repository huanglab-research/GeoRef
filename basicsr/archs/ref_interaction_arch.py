import math
import copy
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Dropout, Module
from typing import  Optional
from einops import rearrange, repeat
from .resnet_fpn_arch import ResNetFPN
from basicsr.utils.registry import ARCH_REGISTRY
class PositionEncodingSine(nn.Module):
    """This is a sinusoidal position encoding that generalized to 2-dimensional
    images."""

    def __init__(self, d_model, max_shape=(1024, 1024)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2).float()
            * (-math.log(10000.0) / d_model // 2)
        )  # bug ?
        # div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        # self.gamma = nn.Parameter(data=torch.ones(1), requires_grad=True)
        pe[0::4, :, :] = torch.sin(x_position * div_term)  #  *(training_dim/test_dim)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, : x.size(2), : x.size(3)]


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        # pdb.set_trace()
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum('nshd,nshv->nhdv', K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum('nlhd,nhd->nlh', Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum('nlhd,nhdv,nlh->nlhv', Q, KV,
                                      Z) * v_length

        return queried_values.contiguous()


class MultiHeadAttention(Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        kdim=None,
        vdim=None,
        attention='linear',
    ):
        super(MultiHeadAttention, self).__init__()
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        # self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.nhead = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        assert (self.head_dim * num_heads == self.embed_dim
                ), 'embed_dim must be divisible by num_heads'

        # multi-head attention
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.attention = LinearAttention(
        ) if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, q, k, v, q_mask=None, kv_mask=None):
        bs = q.size(0)
        # multi-head attention
        query = self.q_proj(q).view(bs, -1, self.nhead,
                                    self.head_dim)  # [N, L, (H, D)]
        key = self.k_proj(k).view(bs, -1, self.nhead,
                                  self.head_dim)  # [N, S, (H, D)]
        value = self.v_proj(v).view(bs, -1, self.nhead, self.head_dim)
        message = self.attention(query,
                                 key,
                                 value,
                                 q_mask=q_mask,
                                 kv_mask=kv_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead *
                                          self.head_dim))  # [N, L, C]

        return message


class FullAttention(Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-head scaled dot-product attention, a.k.a full attention.

        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum('nlhd,nshd->nlsh', queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(
                ~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]),
                float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1.0 / queries.size(3)**0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum('nlsh,nshd->nlhd', A, values)

        return queried_values.contiguous()


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, attention="full"):
        super(EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = (
            LinearAttention()
            if attention == "linear"
            else FullAttention()
        )
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2, bias=False),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.pre_norm_q = nn.LayerNorm(d_model)
        self.pre_norm_kv = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        # pdb.set_trace()
        bs = x.size(0)
        query, key, value = (
            self.pre_norm_q(x),
            self.pre_norm_kv(source),
            self.pre_norm_kv(source),
        )

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(
            query, key, value, q_mask=x_mask, kv_mask=source_mask
        )  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        # message = self.norm1(message)

        # feed-forward network
        x = x + message
        message2 = self.mlp(self.norm2(x))

        return x + message2


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.multihead_attn = MultiHeadAttention(d_model, nhead)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_pos=None, m_pos=None
    ):
        """
        Args:
            tgt (torch.Tensor): [N, L, C]
            memory (torch.Tensor): [N, S, C]
            tgt_mask (torch.Tensor): [N, L] (optional)
            memory_mask (torch.Tensor): [N, S] (optional)
        """
        # pdb.set_trace()
        bs = tgt.size(0)
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, tgt_pos)
        tgt2 = self.self_attn(q, k, v=tgt2, q_mask=tgt_mask, kv_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            q=self.with_pos_embed(tgt2, tgt_pos),
            k=self.with_pos_embed(memory, m_pos),
            v=memory,
            q_mask=tgt_mask,
            kv_mask=memory_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.mlp(tgt2)
        tgt = tgt + tgt2

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_pos: Optional[Tensor] = None,
        m_pos: Optional[Tensor] = None,
    ):
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_pos=tgt_pos,
                m_pos=m_pos,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


def make_head_layer(cnv_dim, curr_dim, out_dim, head_name=None):

    fc = nn.Sequential(
        nn.Conv2d(cnv_dim, curr_dim, kernel_size=3, padding=1, bias=True),
        # nn.BatchNorm2d(curr_dim, eps=1e-3, momentum=0.01),
        nn.ReLU(inplace=True),
        nn.Conv2d(curr_dim, out_dim, kernel_size=3, stride=1, padding=1),
    )

    for l in fc.modules():
        if isinstance(l, nn.Conv2d):
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    return fc


class SegmentationModule(nn.Module):
    def __init__(self, d_model, num_query):
        super(SegmentationModule, self).__init__()
        self.num_query = num_query
        self.block = make_head_layer(
            d_model, d_model // 2, 1, head_name="classification"
        )
        # self.bn = nn.BatchNorm2d(1, eps=1e-3, momentum=0.01)
        # self.gamma = nn.Parameter(data=torch.ones(1, ), requires_grad=True)

    def forward(self, x, hs, mask=None):
        # x:[n, 256, h, w]  hs:[n, num_q, 256]

        if mask is not None:
            attn_mask = torch.einsum("mqc,mchw->mqhw", hs, x)
            attn_mask = attn_mask.sigmoid() * mask.unsqueeze(1)
            classification = self.block(x * attn_mask + x).sigmoid().squeeze(1) * mask
        else:
            attn_mask = torch.einsum("mqc,mchw->mqhw", hs, x)
            attn_mask = attn_mask.sigmoid()
            classification = self.block(x * attn_mask + x).sigmoid().squeeze(1)
        return classification


class InteractionModule(nn.Module):

    def __init__(self, d_model=256, nhead=8, attention="linear"):
        super(InteractionModule, self).__init__()
        self.d_model = d_model
        self.num_query = 1
        self.cas_module = SegmentationModule(d_model, self.num_query)

        encoder_layer = EncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            attention=attention,
        )
        self.layer_names1 = [
            "self",
            "cross",
        ]
        self.layers1 = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names1))]
        )

        self.feature_embed = nn.Embedding(self.num_query, self.d_model)
        decoder_layer = DecoderLayer(
            d_model,
            nhead,
            dropout=0.1,
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=2)
        self.layer_names2 = ["cross"]
        self.layers2 = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names2))]
        )

        self.layer_names3 = [
            "self",
            "cross",
        ]
        self.layers3 = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names3))]
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def transformer(self, x0, x1, x0_mask, x1_mask, layer_name, layer):
        if layer_name == "self":
            src0, src1 = x0, x1
            src0_mask, src1_mask = x0_mask, x1_mask
        elif layer_name == "cross":
            src0, src1 = x1, x0
            src0_mask, src1_mask = x1_mask, x0_mask
        else:
            raise KeyError
        if (
            x0.shape == x1.shape
            and src0.shape == src1.shape
            and x0_mask is not None
            and x1_mask is not None
            and src0_mask is not None
            and src1_mask is not None
            and not self.training
            and 0
        ):
            temp_x = layer(
                torch.cat([x0, x1], dim=0),
                torch.cat([src0, src1], dim=0),
                torch.cat([x0_mask, x1_mask], dim=0),
                torch.cat([src0_mask, src1_mask], dim=0),
            )
            x0, x1 = temp_x.split(x0.shape[0])
        else:
            x0 = layer(x0, src0, x0_mask, src0_mask)
            x1 = layer(x1, src1, x1_mask, src1_mask)
        return x0, x1

    def feature_interaction(self, x0, x1, x0_mask=None, x1_mask=None):
        """x (torch.Tensor): [N, L, C] source (torch.Tensor): [N, S, C] x_mask
        (torch.Tensor): [N, H0, W0]       -> # [N, L] (optional) source_mask
        (torch.Tensor): [N, H1, W1]  -> # [N, S] (optional)"""
        bs = x0.size(0)
        assert self.d_model == x0.size(
            2
        ), "the feature number of src and transformer must be equal"
        if x0_mask != None and x1_mask != None:
            x0_mask, x1_mask = x0_mask.flatten(-2), x1_mask.flatten(-2)

        # stage 1
        for i, (layer, name) in enumerate(zip(self.layers1, self.layer_names1)):
            x0, x1 = self.transformer(x0, x1, x0_mask, x1_mask, name, layer)

        # stage 2
        feature_embed0 = self.feature_embed.weight.unsqueeze(0).repeat(
            bs, 1, 1
        )  # [bs, num_q, c]
        feature_embed1 = self.feature_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        tgt0 = torch.zeros_like(feature_embed0)
        tgt1 = torch.zeros_like(feature_embed1)

        if (
            0
        ):  # x0.shape==x1.shape and x0_mask is not None and x0_mask.shape==x1_mask.shape:
            hs_o = self.decoder(
                torch.cat([tgt0, tgt1], dim=0),
                torch.cat([x0, x1], dim=0),
                tgt_mask=None,
                memory_mask=torch.cat([x0_mask, x1_mask], dim=0),
                tgt_pos=torch.cat([feature_embed0, feature_embed1], dim=0),
            )
            hs0, hs1 = hs_o.split(bs)
        else:
            hs0 = self.decoder(
                tgt0, x0, tgt_mask=None, memory_mask=x0_mask, tgt_pos=feature_embed0
            )
            hs1 = self.decoder(
                tgt1, x1, tgt_mask=None, memory_mask=x1_mask, tgt_pos=feature_embed1
            )

        for i, (layer, name) in enumerate(zip(self.layers2, self.layer_names2)):
            if not self.training and x0.shape == x1.shape and x0_mask is not None:
                x_, hs_ = self.transformer(
                    torch.cat([x0, x1], dim=0),
                    torch.cat([hs1, hs0], dim=0),
                    torch.cat([x0_mask, x1_mask], dim=0),
                    None,
                    name,
                    layer,
                )
                x0, x1 = x_.split(bs)
                hs1, hs0 = hs_.split(bs)
            else:
                x0, hs1 = self.transformer(x0, hs1, x0_mask, None, name, layer)
                x1, hs0 = self.transformer(x1, hs0, x1_mask, None, name, layer)

        x0_mid = x0
        x1_mid = x1
        # stage 3
        for i, (layer, name) in enumerate(zip(self.layers3, self.layer_names3)):
            x0, x1 = self.transformer(x0, x1, x0_mask, x1_mask, name, layer)

        return x0, x1, hs0, hs1, x0_mid, x1_mid

    def covisible_segment(self, x0_mid, x1_mid, hs0, hs1, x0_mask, x1_mask):
        bs = x0_mid.size(0)
        if (
            x0_mask is not None
            and x1_mask is not None
            and x0_mask.shape == x1_mask.shape
            and not self.training
        ):
            cas_scores = self.cas_module(
                torch.cat([x0_mid, x1_mid], dim=0),
                torch.cat([hs0, hs1], dim=0),
                torch.cat([x0_mask, x1_mask], dim=0),
            )
            cas_score0, cas_score1 = cas_scores.split(bs)
        elif x0_mid.shape == x1_mid.shape and x0_mask is None and not self.training:
            cas_scores = self.cas_module(
                torch.cat([x0_mid, x1_mid], dim=0),
                torch.cat([hs0, hs1], dim=0),
            )
            cas_score0, cas_score1 = cas_scores.split(bs)
        else:
            cas_score0 = self.cas_module(x0_mid, hs0, x0_mask)
            cas_score1 = self.cas_module(x1_mid, hs1, x1_mask)
        return cas_score0, cas_score1

    def forward(self, x0, x1, x0_mask=None, x1_mask=None, use_cas=True):


        h0, w0 = x0.shape[2:]
        h1, w1 = x1.shape[2:]
        bs = x0.shape[0]
        x0 = rearrange(x0, "n c h w -> n (h w) c")
        x1 = rearrange(x1, "n c h w -> n (h w) c")
        out0, out1, hs0, hs1, x0_mid, x1_mid = self.feature_interaction(
            x0, x1, x0_mask, x1_mask
        )

        if use_cas:
            x0_mid = rearrange(x0_mid, "n (h w) c -> n c h w", h=h0, w=w0).contiguous()
            x1_mid = rearrange(x1_mid, "n (h w) c -> n c h w", h=h1, w=w1).contiguous()

            cas_score0, cas_score1 = self.covisible_segment(
                x0_mid, x1_mid, hs0, hs1, x0_mask, x1_mask
            )
        else:
            cas_score0, cas_score1 = torch.ones((bs, h0, w0)).to(x0), torch.ones(
                (bs, h1, w1)
            ).to(x1)

        return out0, out1, cas_score0, cas_score1
@ARCH_REGISTRY.register()
class RefInteractionNet(nn.Module):

    def __init__(self, d_model=256, initial_dim=128, block_dims=[128, 196, 256], resolution=(32, 8, 2)):
        super(RefInteractionNet, self).__init__()
        self.pos_encoding = PositionEncodingSine(
            d_model, max_shape=(512, 512))
        self.backbone = ResNetFPN(initial_dim, block_dims, resolution)
        self.feature_interaction = InteractionModule(d_model)

    def forward(self, match_img_in, img_ref):
        feat_d8_in = self.backbone(match_img_in)
        feat_d8_ref = self.backbone(img_ref)
        feat_d8_in = self.pos_encoding(feat_d8_in)
        feat_d8_ref = self.pos_encoding(feat_d8_ref)
        mask_feat_in, mask_feat_ref, co_visible_in, co_visible_ref = self.feature_interaction(feat_d8_in, feat_d8_ref)

        co_visible_in = F.interpolate(co_visible_in.unsqueeze(1), scale_factor=2, mode='bicubic') # [n*batch, 1, h, w]
        co_visible_ref = F.interpolate(co_visible_ref.unsqueeze(1), scale_factor=2, mode='bicubic')
        co_visible = torch.cat((co_visible_in, co_visible_ref), dim=1)
        # co_visible = co_visible.view(batch, n, 2, h*2, w*2)

        return co_visible

        return {
            'co_visible_in': co_visible_in,
            'co_visible_ref': co_visible_ref
        }
