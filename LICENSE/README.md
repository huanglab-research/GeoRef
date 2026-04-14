# License and Acknowledgement

This BasicSR project is released under the Apache 2.0 license.

- StyleGAN2
  - The codes are modified from the repository [stylegan2-pytorch](XXXX). Many thanks to the author - [Kim Seonghyeon](XXXX)  :blush: for translating from the official TensorFlow codes to PyTorch ones. Here is the [license](LICENSE-stylegan2-pytorch) of stylegan2-pytorch.
  - The official repository is XXXX, and here is the [NVIDIA license](./LICENSE-NVIDIA).
- DFDNet
  - The codes are largely modified from the repository [DFDNet](XXXX). Their license is [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](XXXX).
- DiffJPEG
  - Modified from XXXX.
- [pytorch-image-models](XXXX)
  - We use the implementation of `DropPath` and `trunc_normal_` from [pytorch-image-models](XXXX). The LICENSE is included as [LICENSE_pytorch-image-models](LICENSE/LICENSE_pytorch-image-models).
- [SwinIR](XXXX)
  - The arch implementation of SwinIR is from [SwinIR](XXXX). The LICENSE is included as [LICENSE_SwinIR](LICENSE/LICENSE_SwinIR).
- [ECBSR](XXXX)
  - The arch implementation of ECBSR is from [ECBSR](XXXX). The LICENSE of ECBSR is [Apache License 2.0](XXXX)

## References

1. NIQE metric: the codes are translated from the [official MATLAB codes](XXXX)

    > A. Mittal, R. Soundararajan and A. C. Bovik, "Making a Completely Blind Image Quality Analyzer", IEEE Signal Processing Letters, 2012.

1. FID metric: the codes are modified from [pytorch-fid](XXXX) and [stylegan2-pytorch](XXXX).
