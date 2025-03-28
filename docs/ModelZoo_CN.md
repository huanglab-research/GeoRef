# 模型库和基准

[English](ModelZoo.md) **|** [简体中文](ModelZoo_CN.md)

:arrow_double_down: 百度网盘: [预训练模型](XXXX) **|** [复现实验](XXXX)
:arrow_double_down: Google Drive: [Pretrained Models](XXXX) **|** [Reproduced Experiments](XXXX)

---

我们提供了:

1. 官方的模型, 它们是从官方release的models直接转化过来的
1. 复现的模型, 使用`BasicSR`的框架复现的, 提供模型和log的例子

下载的模型可以放在 `experiments/pretrained_models` 文件夹.

**[下载官方提供的预训练模型]** ([Google Drive](XXXX), [百度网盘](XXXX))
你可以使用以下脚本从Google Drive下载预训练模型.

```python
python scripts/download_pretrained_models.py ESRGAN
# method can be ESRGAN, EDVR, StyleGAN, EDSR, DUF, DFDNet, dlib
```

**[下载复现的模型和log]** ([Google Drive](XXXX), [百度网盘](XXXX))

此外, 我们在 [wandb](XXXX) 上更新了模型训练的过程和曲线. 大家可以方便的比较:

**[wandb训练曲线](XXXX)**

<p align="center">
<a href="XXXX" target="_blank">
   <img src="../assets/wandb.jpg" height="350">
</a></p>

#### 目录

1. [图像超分辨率](#图像超分辨率)
    1. [图像超分官方模型](#图像超分官方模型)
    1. [图像超分复现模型](#图像超分复现模型)
1. [视频超分辨率](#视频超分辨率)

## 图像超分辨率

在计算指标时:

- 所有的图像各条边crop了scale的像素
- 都在RGB通道上测试

### 图像超分官方模型

|Exp Name         | Set5 (PSNR/SSIM)     | Set14 (PSNR/SSIM)   |DIV2K100 (PSNR/SSIM)   |
| :------------- | :----------:    | :----------:   |:----------:   |
| EDSR_Mx2_f64b16_DIV2K_official-3ba7b086 | 35.7768 / 0.9442 | 31.4966 / 0.8939 | 34.6291 / 0.9373 |
| EDSR_Mx3_f64b16_DIV2K_official-6908f88a | 32.3597 / 0.903 | 28.3932 / 0.8096 | 30.9438 / 0.8737 |
| EDSR_Mx4_f64b16_DIV2K_official-0c287733 | 30.1821 / 0.8641 | 26.7528 / 0.7432 | 28.9679 / 0.8183 |
| EDSR_Lx2_f256b32_DIV2K_official-be38e77d | 35.9979 / 0.9454 | 31.8583 / 0.8971 | 35.0495 / 0.9407 |
| EDSR_Lx3_f256b32_DIV2K_official-3660f70d | 32.643 / 0.906 | 28.644 / 0.8152 | 31.28 / 0.8798 |
| EDSR_Lx4_f256b32_DIV2K_official-76ee1c8f | 30.5499 / 0.8701 | 27.0011 / 0.7509 | 29.277 / 0.8266 |

### 图像超分复现模型

实验名称的命名规则参见 [Config_CN.md](Config_CN.md).

|Exp Name         | Set5 (PSNR/SSIM)     | Set14 (PSNR/SSIM)   |DIV2K100 (PSNR/SSIM)   |
| :------------- | :----------:    | :----------:   |:----------:   |
| 001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb | 30.2468 / 0.8651 | 26.7817 / 0.7451 | 28.9967 / 0.8195 |
| 002_MSRResNet_x2_f64b16_DIV2K_1000k_B16G1_001pretrain_wandb | 35.7483 / 0.9442 | 31.5403 / 0.8937 |34.6699 / 0.9377|
| 003_MSRResNet_x3_f64b16_DIV2K_1000k_B16G1_001pretrain_wandb | 32.4038 / 0.9032| 28.4418 / 0.8106|30.9726 / 0.8743 |
| 004_MSRGAN_x4_f64b16_DIV2K_400k_B16G1_wandb | 28.0158 / 0.8087|24.7474 / 0.6623 | 26.6504 / 0.7462|
| | | | |
| 201_EDSR_Mx2_f64b16_DIV2K_300k_B16G1_wandb | 35.7395 / 0.944|31.4348 / 0.8934 |34.5798 / 0.937 |
| 202_EDSR_Mx3_f64b16_DIV2K_300k_B16G1_201pretrain_wandb|32.315 / 0.9026 |28.3866 / 0.8088 |30.9095 / 0.8731|
| 203_EDSR_Mx4_f64b16_DIV2K_300k_B16G1_201pretrain_wandb|30.1726 / 0.8641 |26.721 / 0.743 |28.9506 / 0.818|
| 204_EDSR_Lx2_f256b32_DIV2K_300k_B16G1_wandb | 35.9792 / 0.9453 | 31.7284 / 0.8959 | 34.9544 / 0.9399 |
| 205_EDSR_Lx3_f256b32_DIV2K_300k_B16G1_204pretrain_wandb | 32.6467 / 0.9057 | 28.6859 / 0.8152 | 31.2664 / 0.8793 |
| 206_EDSR_Lx4_f256b32_DIV2K_300k_B16G1_204pretrain_wandb | 30.4718 / 0.8695 | 26.9616 / 0.7502 | 29.2621 / 0.8265 |

## 视频超分辨率

#### Evaluation

In the evaluation, we include all the input frames and do not crop any border pixels unless otherwise stated.<br/>
We do not use the self-ensemble (flip testing) strategy and any other post-processing methods.

## EDVR

**Name convention**<br/>
EDVR\_(training dataset)\_(track name)\_(model complexity)

- track name. There are four tracks in the NTIRE 2019 Challenges on Video Restoration and Enhancement:
    - **SR**: super-resolution with a fixed downsampling kernel (MATLAB bicubic downsampling kernel is frequently used). Most of the previous video SR methods focus on this setting.
    - **SRblur**: the inputs are also degraded with motion blur.
    - **deblur**: standard deblurring (motion blur).
    - **deblurcomp**: motion blur + video compression artifacts.
- model complexity
    - **L** (Large): # of channels = 128, # of back residual blocks = 40. This setting is used in our competition submission.
    - **M** (Moderate): # of channels = 64, # of back residual blocks = 10.

| Model name |[Test Set] PSNR/SSIM |
|:----------:|:----------:|
| EDVR_Vimeo90K_SR_L | [Vid4] (Y<sup>1</sup>) 27.35/0.8264 [[↓Results]](XXXX)<br/> (RGB) 25.83/0.8077|
| EDVR_REDS_SR_M | [REDS] (RGB) 30.53/0.8699 [[↓Results]](XXXX-)|
| EDVR_REDS_SR_L | [REDS] (RGB) 31.09/0.8800 [[↓Results]](XXXX)|
| EDVR_REDS_SRblur_L | [REDS] (RGB) 28.88/0.8361 [[↓Results]](XXXXd0SXicwFEPZH)|
| EDVR_REDS_deblur_L | [REDS] (RGB) 34.80/0.9487 [[↓Results]](XXXX)|
| EDVR_REDS_deblurcomp_L | [REDS] (RGB) 30.24/0.8567 [[↓Results]](XXXX)  |

<sup>1</sup> Y or RGB denotes the evaluation on Y (luminance) or RGB channels.

#### Stage 2 models for the NTIRE19 Competition

| Model name |[Test Set] PSNR/SSIM |
|:----------:|:----------:|
| EDVR_REDS_SR_Stage2 | [REDS] (RGB) / [[↓Results]]()|
| EDVR_REDS_SRblur_Stage2 | [REDS] (RGB) / [[↓Results]]()|
| EDVR_REDS_deblur_Stage2 | [REDS] (RGB) / [[↓Results]]()|
| EDVR_REDS_deblurcomp_Stage2 | [REDS] (RGB) / [[↓Results]]()  |


## DUF
The models are converted from the [officially released models](XXXX). <br/>

| Model name | [Test Set] PSNR/SSIM<sup>1</sup> | Official Results<sup>2</sup> |
|:----------:|:----------:|:----------:|
| DUF_x4_52L_official<sup>3</sup> | [Vid4] (Y<sup>4</sup>) 27.33/0.8319 [[↓Results]](XXXX)<br/> (RGB) 25.80/0.8138   | (Y) 27.33/0.8318 [[↓Results]](XXXX)<br/> (RGB) 25.79/0.8136 |
| DUF_x4_28L_official | [Vid4]  | |
| DUF_x4_16L_official | [Vid4]  | |
| DUF_x3_16L_official | [Vid4]  | |
| DUF_x2_16L_official | [Vid4]  | |

<sup>1</sup> We crop eight pixels near image boundary for DUF due to its severe boundary effects. <br/>
<sup>2</sup> The official results are obtained by running the official codes and models. <br/>
<sup>3</sup> Different from the official codes, where `zero padding` is used for border frames, we use `new_info` strategy. <br/>
<sup>4</sup> Y or RGB denotes the evaluation on Y (luminance) or RGB channels.

## TOF
The models are converted from the [officially released models](XXXX).<br/>

| Model name | [Test Set] PSNR/SSIM | Official Results<sup>1</sup> |
|:----------:|:----------:|:----------:|
| TOF_official<sup>2</sup> | [Vid4] (Y<sup>3</sup>) 25.86/0.7626 [[↓Results]](XXXX)<br/> (RGB)  24.38/0.7403 | (Y) 25.89/0.7651 [[↓Results]](XXXX)<br/> (RGB)  24.41/0.7428 |

<sup>1</sup> The official results are obtained by running the official codes and models. Note that TOFlow does not provide a strategy for border frame recovery and we simply use a `replicate` strategy for border frames. <br/>
<sup>2</sup> The converted model has slightly different results, due to different implementation. And we use `new_info` strategy for border frames. <br/>
<sup>3</sup> Y or RGB denotes the evaluation on Y (luminance) or RGB channels.
