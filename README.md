# Multiview Detection with Feature Perspective Transformation [[Website](https://hou-yz.github.io/publication/2020-eccv2020-mvdet)] [[arXiv](https://arxiv.org/abs/2007.07247)]

```
@inproceedings{hou2020multiview,
  title={Multiview Detection with Feature Perspective Transformation},
  author={Hou, Yunzhong and Zheng, Liang and Gould, Stephen},
  booktitle={ECCV},
  year={2020}
}
```

Please visit [link](https://github.com/hou-yz/MVDeTr) for our new work MVDeTr, a transformer-powered multiview detector that achieves new state-of-the-art!

## Overview
We release the PyTorch code for **MVDet**, a state-of-the-art multiview pedestrian detector; and **MultiviewX** dataset, a novel synthetic multiview pedestrian detection datatset.

Wildtrack             |  MultiviewX
:-------------------------:|:-------------------------:
![alt text](https://hou-yz.github.io/images/eccv2020_mvdet_wildtrack_demo.gif "Detection results on Wildtrack dataset")  |  ![alt text](https://hou-yz.github.io/images/eccv2020_mvdet_multiviewx_demo.gif "Detection results on MultiviewX dataset")

 
## Content
- [MultiviewX dataset](#multiviewx-dataset)
    * [Download MultiviewX](#download-multiviewx)
    * [Build your own version](#build-your-own-version)
- [MVDet Code](#mvdet-code)
    * [Dependencies](#dependencies)
    * [Data Preparation](#data-preparation)
    * [Training](#training)



## MultiviewX dataset
Using pedestrian models from [PersonX](https://github.com/sxzrt/Dissecting-Person-Re-ID-from-the-Viewpoint-of-Viewpoint), in Unity, we build a novel synthetic dataset **MultiviewX**. 

![alt text](https://hou-yz.github.io/images/eccv2020_mvdet_multiviewx_dataset.jpg "Visualization of MultiviewX dataset")

MultiviewX dataset covers a square of 16 meters by 25 meters. We quantize the ground plane into a 640x1000 grid. There are 6 cameras with overlapping field-of-view in MultiviewX dataset, each of which outputs a 1080x1920 resolution image. We also generate annotations for 400 frames in MultiviewX at 2 fps (same as Wildtrack). On average, 4.41 cameras are covering the same location. 

### Download MultiviewX
Please refer to this [link](https://1drv.ms/u/s!AtzsQybTubHfhYZ9Ghhahbp20OX9kA?e=Hm9Xdg) for download.

### Build your own version
Please refer to this [repo](https://github.com/hou-yz/MultiviewX) for a detailed guide & toolkits you might need.




## MVDet Code
This repo is dedicated to the code for **MVDet**. 

![alt text](https://hou-yz.github.io/images/eccv2020_mvdet_architecture.png "Architecture for MVDet")

### Dependencies
This code uses the following libraries
- python 3.7+
- pytorch 1.4+ & tochvision
- numpy
- matplotlib
- pillow
- opencv-python
- kornia
- wandb & python-dotenv
- matlab & matlabengine (required for evaluation) (see this [link](/multiview_detector/evaluation/README.md) for detailed guide)

### Data Preparation
By default, all datasets are in `~/Data/`. We use [MultiviewX](#multiviewx-dataset) and [Wildtrack](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) in this project. 

On Kaggle, the dataset classes also automatically discover datasets mounted anywhere under
`/kaggle/input` or `/kaggle/working`, so the paths in the training scripts do not need to be changed.
The local `~/Data/` path is still preferred when it exists.
Generated ground-truth files for read-only datasets under `/kaggle/input` are written to
`/kaggle/working/mvdet_cache`. Set `MVDET_CACHE_DIR` to use a different cache directory.

Your `~/Data/` folder should look like this
```
Data
├── MultiviewX/
│   └── ...
└── Wildtrack/ 
    └── ...
```

### Training
The training entrypoint logs config, train/validation metrics, timings, and GPU usage to
the `GFA26AI02/baseline-expriments` Weights & Biases project. Put either
`WANDB_API_KEY=...` (preferred) or `WANDB_API=...` in the repository `.env` file.

Run one command per dataset; each command creates a separate, comparable W&B run:
```shell script
CUDA_VISIBLE_DEVICES=0,1 python main.py -d wildtrack
CUDA_VISIBLE_DEVICES=0,1 python main.py -d multiviewx
``` 

To train on a dataset stored elsewhere (including a partial-annotation copy),
pass its root directly. The dataset type is detected from its calibration and
directory layout, so `-d` is not needed:

```shell script
CUDA_VISIBLE_DEVICES=0,1 python main.py --data_path /path/to/partial-multiviewx
CUDA_VISIBLE_DEVICES=0,1 python main.py --data_path /path/to/partial-wildtrack --loss bev_brl
```

`--data_path` may also point to a parent directory containing one supported
dataset. If both Wildtrack and MultiviewX are found below that directory, point
it to the desired dataset root instead.

This should automatically return evaluation results similar to the reported 88.2\% MODA on Wildtrack dataset. 

### Partial-annotation BEV-BRL

The repository includes an opt-in ground-plane BRL implementation. It keeps
the original MVDet feature projection and fusion path, then applies a
partial-annotation-aware loss to the final ground-plane logits. The loss uses:

- a max-Gaussian target around observed ground-plane points;
- homography-derived camera coverage;
- top-2 consensus from projected per-view foot heatmaps;
- focal loss for easy negatives and high-BEV peaks rejected by view consensus;
- mirror focal loss for high-confidence, multi-view-supported local peaks;
- ignored gradients for cells between the negative and mirror thresholds;
- configurable warm-up and BRL weight ramp-up.

Dataset label dropping is intentionally not performed by the loss. Supply a
dataset whose `map_gt` and per-view annotations have already been made partial,
with each dropped pedestrian removed consistently from all synchronized views.

Run the proposed loss with:

```shell script
CUDA_VISIBLE_DEVICES=0,1 python main.py --data_path /path/to/partial-wildtrack \
  --loss bev_brl \
  --alpha 0.25 \
  --brl_warmup_epochs 3 \
  --brl_ramp_epochs 3 \
  --brl_weight 0.1
```

The existing baseline remains available with `--loss gaussian_mse`. For a
60% person-level annotation drop, the defaults use a BEV mirror threshold of
0.60, a projected-view threshold of 0.55, at least two supporting cameras, and
at most 1.5 mirror peaks per observed ground-plane point. Use
`--brl_no_consensus` for an ablation based only on the fused BEV confidence.

BRL component losses, mask sizes, current ramp weight, maximum BEV
probability, and maximum view consensus are included in the trainer metrics and
W&B logs.

### Pre-trained models
You can download the checkpoints at this [link](https://1drv.ms/u/s!AtzsQybTubHfhNRE9Iy8IjsGMXB17A?e=CCqhIQ).
