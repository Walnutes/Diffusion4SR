# Diffusion4MosaicSR
Super Resolution Utilizing the Denoising Diffusion Probabilistic Models
From this [URL](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement) ,We train new model for the task of demosaic the text  synthesized through this [tool](https://github.com/clovaai/synthtiger).

## Experiments

Lots of experiments are provided in the folder:experiments/results including the Super Resoulution/Low Resolution/High Resolution(usually also GroundTruth)，the process of denoising and the comparations between different fonds,styles text given the model.

<div align="center">

### SynthTIGER 🐯 : Synthetic Text Image Generator

[![PyPI version](https://img.shields.io/pypi/v/synthtiger)](https://pypi.org/project/synthtiger/)
[![CI](https://github.com/clovaai/synthtiger/actions/workflows/ci.yml/badge.svg)](https://github.com/clovaai/synthtiger/actions/workflows/ci.yml)
[![Docs](https://github.com/clovaai/synthtiger/actions/workflows/docs.yml/badge.svg)](https://github.com/clovaai/synthtiger/actions/workflows/docs.yml)
[![License](https://img.shields.io/github/license/clovaai/synthtiger)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Synthetic Text Image Generator for OCR Model | [Paper](https://arxiv.org/abs/2107.09313) | [Documentation](https://clovaai.github.io/synthtiger/) | [Datasets](#datasets)

</div>

<img src="https://user-images.githubusercontent.com/12423224/153699080-29da7908-0662-4435-ba27-dd07c3bbb7f2.png"/>

## Usage
### Environment
```python
pip install -r requirement.txt
```

### Pretrained Model

TODO: Addition Needed!

### Data Prepare

#### New Start

If you didn't have the data, you can prepare it by following steps:

- [FFHQ 128×128](https://github.com/NVlabs/ffhq-dataset) | [FFHQ 512×512](https://www.kaggle.com/arnaud58/flickrfaceshq-dataset-ffhq)
- [CelebaHQ 256×256](https://www.kaggle.com/badasstechie/celebahq-resized-256x256) | [CelebaMask-HQ 1024×1024](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view)

Download the dataset and prepare it in **LMDB** or **PNG** format using script.

```python
# Resize to get 16×16 LR_IMGS and 128×128 HR_IMGS, then prepare 128×128 Fake SR_IMGS by bicubic interpolation
python data/prepare_data.py  --path [dataset root]  --out [output root] --size 16,128 -l
```

then you need to change the datasets config to your data path and image resolution: 

```json
"datasets": {
    "train": {
        "dataroot": "dataset/ffhq_16_128", // [output root] in prepare.py script
        "l_resolution": 16, // low resolution need to super_resolution
        "r_resolution": 128, // high resolution
        "datatype": "lmdb", //lmdb or img, path of img files
    },
    "val": {
        "dataroot": "dataset/celebahq_16_128", // [output root] in prepare.py script
    }
},
```

#### Own Data

You also can use your image data by following steps, and we have some examples in dataset folder.

At first, you should organize the images layout like this, this step can be finished by `data/prepare_data.py` automatically:

```shell
# set the high/low resolution images, bicubic interpolation images path 
dataset/celebahq_16_128/
├── hr_128 # it's same with sr_16_128 directory if you don't have ground-truth images.
├── lr_16 # vinilla low resolution images
└── sr_16_128 # images ready to super resolution
```

```python
# super resolution from 16 to 128
python data/prepare_data.py  --path [dataset root]  --out celebahq --size 16,128 -l
```

*Note: Above script can be used whether you have the vanilla high-resolution images or not.*

then you need to change the dataset config to your data path and image resolution: 

```json
"datasets": {
    "train|val": { // train and validation part
        "dataroot": "dataset/celebahq_16_128",
        "l_resolution": 16, // low resolution need to super_resolution
        "r_resolution": 128, // high resolution
        "datatype": "img", //lmdb or img, path of img files
    }
},
```

### Training/Resume Training

```python
# Use sr.py and sample.py to train the super resolution task and unconditional generation task, respectively.
# Edit json files to adjust network structure and hyperparameters
python sr.py -p train -c config/xxx_train.json
```

### Test/Evaluation

```python
# Edit json to add pretrain model path and run the evaluation 
python sr.py -p val -c config/xxx_test/val.json

# Quantitative evaluation alone using SSIM/PSNR metrics on given result root
python eval.py -p [result root]
```

### Inference Alone

Set the  image path like steps in `Own Data`, then run the script:

```python
# run the script
python infer.py -c [config file]
```
