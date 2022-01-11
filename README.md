<img src="CIoU.png" width="800px"/>

### English | [简体中文](README_zh-CN.md)

## Complete-IoU Loss and Cluster-NMS for Improving Object Detection and Instance Segmentation. 

Our paper is accepted by **IEEE Transactions on Cybernetics (TCYB)**.

### This repo is based on YOLACT++.

This is the code for our papers:
 - [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/abs/1911.08287)
 - [Enhancing Geometric Factors into Model Learning and Inference for Object Detection and Instance Segmentation](https://arxiv.org/abs/2005.03572)

```
@Inproceedings{zheng2020diou,
  author    = {Zheng, Zhaohui and Wang, Ping and Liu, Wei and Li, Jinze and Ye, Rongguang and Ren, Dongwei},
  title     = {Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression},
  booktitle = {The AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2020},
}

@Article{zheng2021ciou,
  author    = {Zheng, Zhaohui and Wang, Ping and Ren, Dongwei and Liu, Wei and Ye, Rongguang and Hu, Qinghua and Zuo, Wangmeng},
  title     = {Enhancing Geometric Factors in Model Learning and Inference for Object Detection and Instance Segmentation},
  booktitle = {IEEE Transactions on Cybernetics},
  year      = {2021},
}
```

## Description of Cluster-NMS and Its Usage

An example diagram of our Cluster-NMS, where X denotes IoU matrix which is calculated by `X=jaccard(boxes,boxes).triu_(diagonal=1) > nms_thresh` after sorted by score descending. (Here use 0,1 for visualization.)

<img src="cluster-nms01.png" width="1150px"/>
<img src="cluster-nms02.png" width="1150px"/>

The inputs of NMS are `boxes` with size [n,4] and `scores` with size [80,n]. (take coco as example)

There are two ways for NMS. One is that all classes have the same number of boxes. First, we use top k=200 to select the top 200 detections for every class. Then `boxes` will be [80,200,4]. Do Cluster-NMS and keep the boxes with `scores>0.01`. Finally, return top 100 boxes across all classes.

The other approach is that different classes have different numbers of boxes. First, we use a score threshold (e.g. 0.01) to filter out most low score detection boxes. It results in the number of remaining boxes in different classes may be different. Then put all the boxes together and sorted by score descending. (Note that the same box may appear more than once, because its scores of multiple classes are greater than the threshold 0.01.) Adding offset for all the `boxes` according to their class labels. (use `torch.arange(0,80)`.) For example, since the coordinates (x1,y1,x2,y2) of all the boxes are on interval (0,1). By adding offset, if a box belongs to class 61, its coordinates will on interval (60,61). After that, the IoU of boxes belonging to different classes will be 0. (because they are treated as different clusters.) Do Cluster-NMS and return top 100 boxes across all classes. (For this method, please refer to another our repository https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/detection/detection.py)

## Getting Started

### 1) New released! CIoU and Cluster-NMS

1. YOLACT (See [YOLACT](https://github.com/Zzh-tju/CIoU#YOLACT))

2. YOLOv3-pytorch [https://github.com/Zzh-tju/ultralytics-YOLOv3-Cluster-NMS](https://github.com/Zzh-tju/ultralytics-YOLOv3-Cluster-NMS)

3. YOLOv5 (Support batch mode Cluster-NMS. It will speed up NMS when turning on test-time augmentation like multi-scale testing.) [https://github.com/Zzh-tju/yolov5](https://github.com/Zzh-tju/yolov5)

4. SSD-pytorch [https://github.com/Zzh-tju/DIoU-SSD-pytorch](https://github.com/Zzh-tju/DIoU-SSD-pytorch)

### 2) DIoU and CIoU losses into Detection Algorithms
DIoU and CIoU losses are incorporated into state-of-the-art detection algorithms, including YOLO v3, SSD and Faster R-CNN. 
The details of implementation and comparison can be respectively found in the following links. 

1. YOLO v3 [https://github.com/Zzh-tju/DIoU-darknet](https://github.com/Zzh-tju/DIoU-darknet)

2. SSD [https://github.com/Zzh-tju/DIoU-SSD-pytorch](https://github.com/Zzh-tju/DIoU-SSD-pytorch)

3. Faster R-CNN [https://github.com/Zzh-tju/DIoU-pytorch-detectron](https://github.com/Zzh-tju/DIoU-pytorch-detectron)

4. Simulation Experiment [https://github.com/Zzh-tju/DIoU](https://github.com/Zzh-tju/DIoU)

# YOLACT

### Codes location and options

Please take a look at `ciou` function of [layers/modules/multibox_loss.py](layers/modules/multibox_loss.py) for our CIoU loss implementation in PyTorch.

Currently, NMS surports two modes: (See [eval.py](eval.py))

1. Cross-class mode, which ignores classes. (`cross_class_nms=True`, faster than per-class mode but with a slight performance drop.)

2. Per-class mode. (`cross_class_nms=False`)

Currently, NMS supports `fast_nms`, `cluster_nms`, `cluster_diounms`, `spm`, `spm_dist`, `spm_dist_weighted`. 

See [layers/functions/detection.py](layers/functions/detection.py) for our Cluster-NMS implementation in PyTorch.

# Installation

In order to use YOLACT++, make sure you compile the DCNv2 code.

 - Clone this repository and enter it:
   ```Shell
   git clone https://github.com/Zzh-tju/CIoU.git
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

# Evaluation
Here are our YOLACT models (released on May 5th, 2020) along with their FPS on a GTX 1080 Ti and mAP on `coco 2017 val`:

The training is carried on two GTX 1080 Ti with command:
`
python train.py --config=yolact_base_config --batch_size=8
`

| Image Size | Backbone  | Loss  | NMS  | FPS  | box AP  | mask AP  | Weights   |                                                       
|:----:|:-------------:|:-------:|:----:|:----:|:----:|:----:|----------------------------------------------------------------------------------------------------------------------|
| 550  | Resnet101-FPN | SL1  | Fast NMS | 30.6 | 31.5 | 29.1 |[SL1.pth](https://share.weiyun.com/5N840Hm)  | 
| 550  | Resnet101-FPN | CIoU | Fast NMS | 30.6 | 32.1 | 29.6 | [CIoU.pth](https://share.weiyun.com/5EtJ4dJ) | 

To evalute the model, put the corresponding weights file in the `./weights` directory and run one of the following commands. The name of each config is everything before the numbers in the file name (e.g., `yolact_base` for `yolact_base_54_800000.pth`).
## Quantitative Results on COCO
```
# Quantitatively evaluate a trained model on the entire validation set. Make sure you have COCO downloaded as above.

# Output a COCOEval json to submit to the website or to use the run_coco_eval.py script.
# This command will create './results/bbox_detections.json' and './results/mask_detections.json' for detection and instance segmentation respectively.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --output_coco_json

# You can run COCOEval on the files created in the previous command. The performance should match my implementation in eval.py.
python run_coco_eval.py

# To output a coco json file for test-dev, make sure you have test-dev downloaded from above and go
python eval.py --trained_model=weights/yolact_base_54_800000.pth --output_coco_json --dataset=coco2017_testdev_dataset
```
## Qualitative Results on COCO
```
# Display qualitative results on COCO. From here on I'll use a confidence threshold of 0.15.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --display
```
## Cluster-NMS Using Benchmark on COCO
```
python eval.py --trained_model=weights/yolact_base_54_800000.pth --benchmark
```
#### Hardware
 - 1 GTX 1080 Ti
 - Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz

| Image Size | Backbone  | Loss  | NMS  | FPS  | box AP | box AP75 | box AR100 | mask AP | mask AP75 | mask AR100 |
|:----:|:-------------:|:-------:|:------------------------------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 550  | Resnet101-FPN | CIoU  |                 Fast NMS               |**30.6**|  32.1  |  33.9  |  43.0  |  29.6  |  30.9  |  40.3  |
| 550  | Resnet101-FPN | CIoU  |               Original NMS             |  11.5  |  32.5  |  34.1  |  45.1  |  29.7  |  31.0  |  41.7  |
| 550  | Resnet101-FPN | CIoU  |               Cluster-NMS              |  28.8  |  32.5  |  34.1  |  45.2  |  29.7  |  31.0  |  41.7  |
| 550  | Resnet101-FPN | CIoU  |             SPM Cluster-NMS            |  28.6  |  33.1  |  35.2  |  48.8  |**30.3**|**31.7**|  43.6  |
| 550  | Resnet101-FPN | CIoU  |       SPM + Distance Cluster-NMS       |  27.1  |  33.2  |  35.2  |**49.2**|  30.2  |**31.7**|**43.8**|
| 550  | Resnet101-FPN | CIoU  | SPM + Distance + Weighted Cluster-NMS  |  26.5  |**33.4**|**35.5**|  49.1  |**30.3**|  31.6  |**43.8**|

The following table is evaluated by using their pretrained weight of YOLACT. ([yolact_resnet50_54_800000.pth](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EUVpxoSXaqNIlssoLKOEoCcB1m0RpzGq_Khp5n1VX3zcUw))

| Image Size | Backbone  | Loss  | NMS  | FPS  | box AP | box AP75 | box AR100 | mask AP | mask AP75 | mask AR100 |
|:----:|:-------------:|:-------:|:-----------------------------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 550  | Resnet50-FPN | SL1  |                 Fast NMS               |**41.6**|  30.2  |  31.9  |  42.0  |  28.0  |  29.1  |  39.4  |
| 550  | Resnet50-FPN | SL1  |               Original NMS             |  12.8  |  30.7  |  32.0  |  44.1  |  28.1  |  29.2  |  40.7  |
| 550  | Resnet50-FPN | SL1  |               Cluster-NMS              |  38.2  |  30.7  |  32.0  |  44.1  |  28.1  |  29.2  |  40.7  |
| 550  | Resnet50-FPN | SL1  |             SPM Cluster-NMS            |  37.7  |  31.3  |  33.2  |  48.0  |**28.8**|**29.9**|  42.8  |
| 550  | Resnet50-FPN | SL1  |       SPM + Distance Cluster-NMS       |  35.2  |  31.3  |  33.3  |  48.2  |  28.7  |**29.9**|  42.9  |
| 550  | Resnet50-FPN | SL1  | SPM + Distance + Weighted Cluster-NMS  |  34.2  |**31.8**|**33.9**|**48.3**|**28.8**|**29.9**|**43.0**|

The following table is evaluated by using their pretrained weight of YOLACT. ([yolact_base_54_800000.pth](https://drive.google.com/file/d/1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_/view?usp=sharing))

| Image Size | Backbone  | Loss  | NMS  | FPS  | box AP | box AP75 | box AR100 | mask AP | mask AP75 | mask AR100 |
|:----:|:-------------:|:-------:|:-----------------------------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 550  | Resnet101-FPN | SL1  |                 Fast NMS               |**30.6**|  32.5  |  34.6  |  43.9  |  29.8  |  31.3  |  40.8  |
| 550  | Resnet101-FPN | SL1  |               Original NMS             |  11.9  |  32.9  |  34.8  |  45.8  |  29.9  |  31.4  |  42.1  |
| 550  | Resnet101-FPN | SL1  |               Cluster-NMS              |  29.2  |  32.9  |  34.8  |  45.9  |  29.9  |  31.4  |  42.1  |
| 550  | Resnet101-FPN | SL1  |             SPM Cluster-NMS            |  28.8  |  33.5  |  35.9  |  49.7  |**30.5**|**32.1**|  44.1  |
| 550  | Resnet101-FPN | SL1  |       SPM + Distance Cluster-NMS       |  27.5  |  33.5  |  35.9  |**50.2**|  30.4  |  32.0  |**44.3**|
| 550  | Resnet101-FPN | SL1  | SPM + Distance + Weighted Cluster-NMS  |  26.7  |**34.0**|**36.6**|  49.9  |**30.5**|  32.0  |**44.3**|

The following table is evaluated by using their pretrained weight of YOLACT++. ([yolact_plus_base_54_800000.pth](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EVQ62sF0SrJPrl_68onyHF8BpG7c05A8PavV4a849sZgEA))

| Image Size | Backbone  | Loss  | NMS  | FPS  | box AP | box AP75 | box AR100 | mask AP | mask AP75 | mask AR100 |
|:----:|:-------------:|:-------:|:-----------------------------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 550  | Resnet101-FPN | SL1  |                 Fast NMS               |**25.1**|  35.8  |  38.7  |  45.5  |  34.4  |  36.8  |  42.6  |
| 550  | Resnet101-FPN | SL1  |               Original NMS             |  10.9  |  36.4  |  39.1  |  48.0  |  34.7  |  37.1  |  44.1  |
| 550  | Resnet101-FPN | SL1  |               Cluster-NMS              |  23.7  |  36.4  |  39.1  |  48.0  |  34.7  |  37.1  |  44.1  |
| 550  | Resnet101-FPN | SL1  |             SPM Cluster-NMS            |  23.2  |  36.9  |  40.1  |  52.8  |**35.0**|  37.5  |**46.3**|
| 550  | Resnet101-FPN | SL1  |       SPM + Distance Cluster-NMS       |  22.0  |  36.9  |  40.2  |**53.0**|  34.9  |  37.5  |**46.3**|
| 550  | Resnet101-FPN | SL1  | SPM + Distance + Weighted Cluster-NMS  |  21.7  |**37.4**|**40.6**|  52.5  |**35.0**|**37.6**|**46.3**|
#### Note:
 - Things we did but did not appear in the paper: SPM + Distance + Weighted Cluster-NMS. Here the box coordinate weighted average is only performed in `IoU> 0.8`. We searched that `IoU>0.5` is not good for YOLACT and `IoU>0.9` is almost same to `SPM + Distance Cluster-NMS`. (Refer to [CAD](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8265304) for the details of Weighted-NMS.)
 
 - The Original NMS implemented by YOLACT is faster than ours, because they firstly use a score threshold (0.05) to get the set of candidate boxes, then do NMS will be faster (taking YOLACT ResNet101-FPN as example, 22 ~ 23 FPS with a slight performance drop). In order to get the same result with our Cluster-NMS, we modify the process of Original NMS.
 
 - Note that Torchvision NMS has the fastest speed, that is owing to CUDA implementation and engineering accelerations (like upper triangular IoU matrix only). However, our Cluster-NMS requires less iterations for NMS and can also be further accelerated by adopting engineering tricks.

 - Currently, Torchvision NMS use IoU as criterion, not DIoU. However, if we directly replace IoU with DIoU in Original NMS, it will costs much more time due to the sequence operation. Now, Cluster-DIoU-NMS will significantly speed up DIoU-NMS and obtain exactly the same result.

 - Torchvision NMS is a function in Torchvision>=0.3, and our Cluster-NMS can be applied to any projects that use low version of Torchvision and other deep learning frameworks as long as it can do matrix operations. **No other import, no need to compile, less iteration, fully GPU-accelerated and better performance**.

## Images
```Shell
# Display qualitative results on the specified image.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --ima


ge=my_image.png

# Process an image and save it to another file.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=input_image.png:output_image.png

# Process a whole folder of images.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --images=path/to/input/folder:path/to/output/folder
```
## Video
```Shell
# Display a video in real-time. "--video_multiframe" will process that many frames at once for improved performance.
# If you want, use "--display_fps" to draw the FPS directly on the frame.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=my_video.mp4

# Display a webcam feed in real-time. If you have multiple webcams pass the index of the webcam you want instead of 0.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=0

# Process a video and save it to another file. This uses the same pipeline as the ones above now, so it's fast!
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=input_video.mp4:output_video.mp4
```
As you can tell, `eval.py` can do a ton of stuff. Run the `--help` command to see everything it can do.
```Shell
python eval.py --help
```
# Training
By default, we train on COCO. Make sure to download the entire dataset using the commands above.
 - To train, grab an imagenet-pretrained model and put it in `./weights`.
   - For Resnet101, download `resnet101_reducedfc.pth` from [here](https://drive.google.com/file/d/1tvqFPd4bJtakOlmn-uIA492g2qurRChj/view?usp=sharing).
   - For Resnet50, download `resnet50-19c8e357.pth` from [here](https://drive.google.com/file/d/1Jy3yCdbatgXa5YYIdTCRrSV0S9V5g1rn/view?usp=sharing).
   - For Darknet53, download `darknet53.pth` from [here](https://drive.google.com/file/d/17Y431j4sagFpSReuPNoFcj9h7azDTZFf/view?usp=sharing).
 - Run one of the training commands below.
   - Note that you can press ctrl+c while training and it will save an `*_interrupt.pth` file at the current iteration.
   - All weights are saved in the `./weights` directory by default with the file name `<config>_<epoch>_<iter>.pth`.
```Shell
# Trains using the base config with a batch size of 8 (the default).
python train.py --config=yolact_base_config

# Trains yolact_base_config with a batch_size of 5. For the 550px models, 1 batch takes up around 1.5 gigs of VRAM, so specify accordingly.
python train.py --config=yolact_base_config --batch_size=5

# Resume training yolact_base with a specific weight file and start from the iteration specified in the weight file's name.
python train.py --config=yolact_base_config --resume=weights/yolact_base_10_32100.pth --start_iter=-1

# Use the help option to see a description of all available command line arguments
python train.py --help
```

## Multi-GPU Support
YOLACT now supports multiple GPUs seamlessly during training:

 - Before running any of the scripts, run: `export CUDA_VISIBLE_DEVICES=[gpus]`
   - Where you should replace [gpus] with a comma separated list of the index of each GPU you want to use (e.g., 0,1,2,3).
   - You should still do this if only using 1 GPU.
   - You can check the indices of your GPUs with `nvidia-smi`.
 - Then, simply set the batch size to `8*num_gpus` with the training commands above. The training script will automatically scale the hyperparameters to the right values.
   - If you have memory to spare you can increase the batch size further, but keep it a multiple of the number of GPUs you're using.
   - If you want to allocate the images per GPU specific for different GPUs, you can use `--batch_alloc=[alloc]` where [alloc] is a comma seprated list containing the number of images on each GPU. This must sum to `batch_size`.
   
## Acknowledgments

Thank you to [Daniel Bolya](https://github.com/dbolya/) for his fork of [YOLACT & YOLACT++](https://github.com/dbolya/yolact), which is an exellent work for real-time instance segmentation.
