# CIoU & Cluster-NMS
Complete-IoU (CIoU) Loss and Cluster-NMS for Object Detection and Instance Segmentation. This is the code for our papers:
 - [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/abs/1911.08287)
 - [Enhancing Geometric Factors into Model Learning and Inference for Object Detection and Instance Segmentation](https://arxiv.org/abs/1904.02689)

```
@inproceedings{zheng2020distance,
  author    = {Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, Dongwei Ren},
  title     = {Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression},
  booktitle = {The AAAI Conference on Artificial Intelligence (AAAI)},
   year      = {2020},
}
```

## Getting Started

### 1) New released! CIoU into YOLACT (See [Installation](https://github.com/Zzh-tju/CIoU#**Y**ou **O**nly **L**ook **A**t **C**oefficien**T**s))


### 2) DIoU and CIoU losses into Detection Algorithms
DIoU and CIoU losses are incorporated into state-of-the-art detection algorithms, including YOLO v3, SSD and Faster R-CNN. 
The details of implementation and comparison can be respectively found in the following links. 

1. YOLO v3 [https://github.com/Zzh-tju/DIoU-darknet](https://github.com/Zzh-tju/DIoU-darknet)

2. SSD [https://github.com/Zzh-tju/DIoU-SSD-pytorch](https://github.com/Zzh-tju/DIoU-SSD-pytorch)

3. Faster R-CNN [https://github.com/Zzh-tju/DIoU-pytorch-detectron](https://github.com/Zzh-tju/DIoU-pytorch-detectron)

# **Y**ou **O**nly **L**ook **A**t **C**oefficien**T**s
```
    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║   
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║   
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║   
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝ 
```

In order to use YOLACT++, make sure you compile the DCNv2 code. (See [Installation](https://github.com/dbolya/yolact#installation))

# Installation
 - Clone this repository and enter it:
   ```Shell
   git clone https://github.com/dbolya/yolact.git
   cd yolact
   ```
 - Set up the environment using one of the following methods:
   - Using [Anaconda](https://www.anaconda.com/distribution/)
     - Run `conda env create -f environment.yml`
   - Manually with pip
     - Set up a Python3 environment (e.g., using virtenv).
     - Install [Pytorch](http://pytorch.org/) 1.0.1 (or higher) and TorchVision.
     - Install some other packages:
       ```Shell
       # Cython needs to be installed before pycocotools
       pip install cython
       pip install opencv-python pillow pycocotools matplotlib 
       ```
 - If you'd like to train YOLACT, download the COCO dataset and the 2014/2017 annotations. Note that this script will take a while and dump 21gb of files into `./data/coco`.
   ```Shell
   sh data/scripts/COCO.sh
   ```
 - If you'd like to evaluate YOLACT on `test-dev`, download `test-dev` with this script.
   ```Shell
   sh data/scripts/COCO_test.sh
   ```
 - If you want to use YOLACT++, compile deformable convolutional layers (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_1.0)).
   Make sure you have the latest CUDA toolkit installed from [NVidia's Website](https://developer.nvidia.com/cuda-toolkit).
   ```Shell
   cd external/DCNv2
   python setup.py build develop
   ```
