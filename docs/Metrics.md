# Metrics

[English](Metrics.md) **|** [简体中文](Metrics_CN.md)

## PSNR and SSIM

## NIQE

## FID

> FID measures the similarity between two datasets of images. It was shown to correlate well with human judgement of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks.
> FID is calculated by computing the [Fréchet distance](XXXX) between two Gaussians fitted to feature representations of the Inception network.

References

- XXXXid
- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](XXXX)
- [Are GANs Created Equal? A Large-Scale Study](XXXX)

### Pre-calculated FFHQ inception feature statistics

Usually, we put the downloaded inception feature statistics in `basicsr/metrics`.

:arrow_double_down: Google Drive: [metrics data](XXXX)
:arrow_double_down: 百度网盘: [评价指标数据](XXXX) <br>

| File Name         | Dataset | Image Shape    | Sample Numbers|
| :------------- | :----------:|:----------:|:----------:|
| inception_FFHQ_256-0948f50d.pth | FFHQ | 256 x 256 | 50,000 |
| inception_FFHQ_512-f7b384ab.pth | FFHQ | 512 x 512 | 50,000 |
| inception_FFHQ_1024-75f195dc.pth | FFHQ | 1024 x 1024 | 50,000 |
| inception_FFHQ_256_stylegan2_pytorch-abba9d31.pth | FFHQ | 256 x 256 | 50,000 |

- All the FFHQ inception feature statistics calculated on the resized 299 x 299 size.
- `inception_FFHQ_256_stylegan2_pytorch-abba9d31.pth` is converted from the statistics in [stylegan2-pytorch](XXXX).
