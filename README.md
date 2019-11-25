# ImageSR-Pytorch
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)][license]

[license]: https://github.com/linchuming/ImageSR-Pytorch/blob/master/LICENSE

### Preparation
- Python 3.6
- Pytorch 1.1.0

### Model
- Use a simple version EDSR for image super-resolution

### Training
- Prepare training dataset. Please download DIV2K dataset:
[link](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)
- Move the training dataset to ./DIV2K_train_HR
- Check the config in config.yaml
- Run the script `python train.py config.yaml`

### Inference
- Check the config in config.yaml
- Run the script `python inference.py config.yaml`

### Reference
- [ImageSR for Tensorflow](https://github.com/linchuming/ImageSR-Tensorflow)