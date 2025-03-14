import torch
import torch.nn.functional as F



def sample_patches(inputs, patch_size=3, stride=1, padding=0):
    """Extract sliding local patches from an input feature tensor.
    The sampled pathes are row-major.

    Args:
        inputs (Tensor): the input feature maps, shape: (c, h, w).
        patch_size (int): the spatial size of sampled patches. Default: 3.
        stride (int): the stride of sampling. Default: 1.

    Returns:
        patches (Tensor): extracted patches, shape: (c, patch_size,
            patch_size, n_patches).
    """

    c, h, w = inputs.shape
    if padding!=0:
        inputs = F.pad(inputs,(padding, padding, padding, padding))
    patches = inputs.unfold(1, patch_size, stride)\
                    .unfold(2, patch_size, stride)\
                    .reshape(c, -1, patch_size, patch_size)\
                    .permute(0, 2, 3, 1)
    return patches

def conv2d_with_einsum(feat_input, batch, patch_size=3, stride=1, padding=1):
    # 输入维度: feat_input [channel, h, w]， batch [channel, 3, 3, n]

    # 将输入特征图进行展开，展平为滑动窗口形式
    feat_input_unfold_1 = F.unfold(feat_input.unsqueeze(0), kernel_size=patch_size, stride=stride, padding=padding)
    feat_input_unfold = sample_patches(feat_input, patch_size, stride, padding)
    feat_input_unfold = feat_input_unfold.reshape(-1, feat_input_unfold.shape[3]).unsqueeze(0)
    # 现在 feat_input_unfold 形状是 [1, channel*3*3, h*w]，表示每个 3x3 窗口展平后的特征

    # 将 batch 特征块展平，形状变为 [n, channel*3*3]
    batch_flat = batch.reshape(batch.shape[3], -1)  # [n, channel*3*3]


    # 使用 einsum 进行点积，等价于卷积操作
    # einsum 表达式含义: 对于 batch_flat 的每个 n 特征块，与 feat_input_unfold 的每个 [channel*3*3] 位置点积
    corr = torch.einsum('nc,bch->bnh', batch_flat, feat_input_unfold)


    # reshape 结果，使其与卷积后的形状一致 [1, n, h, w]
    h, w = feat_input.shape[1], feat_input.shape[2]
    corr = corr.view(1, batch.shape[3], h, w)

    return corr
def feature_match_index(feat_input,
                        feat_ref,
                        patch_size=3,
                        input_stride=1,
                        ref_stride=1,
                        is_norm=True,
                        norm_input=False):
    """Patch matching between input and reference features.

    Args:
        feat_input (Tensor): the feature of input, shape: (c, h, w).
        feat_ref (Tensor): the feature of reference, shape: (c, h, w).
        patch_size (int): the spatial size of sampled patches. Default: 3.
        stride (int): the stride of sampling. Default: 1.
        is_norm (bool): determine to normalize the ref feature or not.
            Default:True.

    Returns:
        max_idx (Tensor): The indices of the most similar patches.
        max_val (Tensor): The correlation values of the most similar patches.
    """

    # patch decomposition, shape: (c, patch_size, patch_size, n_patches)
    patches_ref = sample_patches(feat_ref, patch_size, ref_stride)

    # normalize reference feature for each patch in both channel and
    # spatial dimensions.

    # batch-wise matching because of memory limitation
    _, h, w = feat_input.shape
    batch_size = int(1024.**2 * 512 / (h * w))
    n_patches = patches_ref.shape[-1]

    max_idx, max_val = None, None
    for idx in range(0, n_patches, batch_size):
        batch = patches_ref[..., idx:idx + batch_size]
        if is_norm:
            batch = batch / (batch.norm(p=2, dim=(0, 1, 2)) + 1e-5)
        corr = conv2d_with_einsum(feat_input, batch)
        # corr = F.conv2d(
        #     feat_input.unsqueeze(0),
        #     batch.permute(3, 0, 1, 2),
        #     stride=input_stride)

        max_val_tmp, max_idx_tmp = corr.squeeze(0).max(dim=0)

        if max_idx is None:
            max_idx, max_val = max_idx_tmp, max_val_tmp
        else:
            indices = max_val_tmp > max_val
            max_val[indices] = max_val_tmp[indices]
            max_idx[indices] = max_idx_tmp[indices] + idx

    if norm_input:
        patches_input = sample_patches(feat_input, patch_size, input_stride)
        norm = patches_input.norm(p=2, dim=(0, 1, 2)) + 1e-5
        norm = norm.view(
            int((h - patch_size) / input_stride + 1),
            int((w - patch_size) / input_stride + 1))
        max_val = max_val / norm

    return max_idx, max_val

def feature_match_topk_index(feat_input,
                        feat_ref,
                        topk=5,
                        patch_size=3,
                        padding=0,
                        input_stride=1,
                        ref_stride=1,
                        is_norm=True,
                        norm_input=False,
                        ):  # 新增参数 k
    """Patch matching between input and reference features.

    Args:
        feat_input (Tensor): the feature of input, shape: (c, h, w).
        feat_ref (Tensor): the feature of reference, shape: (c, h, w).
        patch_size (int): the spatial size of sampled patches. Default: 3.
        input_stride (int): the stride of sampling. Default: 1.
        is_norm (bool): determine to normalize the ref feature or not. Default:True.
        k (int): the number of top similar patches to select. Default: 5.

    Returns:
        topk_idx (Tensor): The indices of the top k most similar patches.
        topk_val (Tensor): The correlation values of the top k most similar patches.
    """

    # patch decomposition, shape: (c, patch_size, patch_size, n_patches)
    patches_ref = sample_patches(feat_ref, patch_size, ref_stride, padding)

    # normalize reference feature for each patch in both channel and
    # spatial dimensions.
    _, h, w = feat_input.shape
    batch_size = int(1024.**2 * 512 / (h * w)) # 512MB
    n_patches = patches_ref.shape[-1]

    topk_idx, topk_val = None, None
    for idx in range(0, n_patches, batch_size):
        batch = patches_ref[..., idx:idx + batch_size]
        if is_norm:
            batch = batch / (batch.norm(p=2, dim=(0, 1, 2)) + 1e-5)
        # corr = conv2d_with_einsum(feat_input, batch)
        corr = F.conv2d(
            feat_input.unsqueeze(0),
            batch.permute(3, 0, 1, 2),
            stride=input_stride,
            padding=padding)

        topk_val_tmp, topk_idx_tmp = torch.topk(corr.squeeze(0),topk,dim=0,largest=True,sorted=True)
        if topk_idx is None:
            topk_idx, topk_val = topk_idx_tmp, topk_val_tmp
        else:
            combined_val = torch.cat((topk_val, topk_val_tmp), dim=0)
            combined_idx = torch.cat((topk_idx, topk_idx_tmp + idx), dim=0)
            sub_topk_val, indices = combined_val.topk(topk, dim=0)
            topk_idx = torch.gather(combined_idx, 0, indices)
            topk_val = torch.gather(combined_val, 0, indices)

    if norm_input:
        patches_input = sample_patches(feat_input, patch_size, input_stride, padding)
        norm = patches_input.norm(p=2, dim=(0, 1, 2)) + 1e-5
        norm = norm.view(
            int((h - patch_size + 2*padding) / input_stride + 1),
            int((w - patch_size + 2*padding) / input_stride + 1))
        topk_val = topk_val / norm

    return topk_idx, topk_val
