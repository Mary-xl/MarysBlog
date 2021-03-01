---
layout:     post
title:      "Object detection: from YOLOv3 to YOLOv5 customized -2"
date:       2020-05-10
author:     Mary Li
header-img:
catalog: true
tags:
- deep learning
- object detection
---
_In 2020, I started a new project on video surveillance, where deep learning based techniques like object detection, 
multiple object tracking, human action recognition are heavily used and modified for our own application._

## All you need to know about object detection

After having going through the development of backbones, we can take a look of different object detection frameworks.
[![6CGgoD.png](https://s3.ax1x.com/2021/02/28/6CGgoD.png)](https://imgtu.com/i/6CGgoD)

One common design we can find is the downsampling-to-upsampling structure. More specifically, normal convolutional layers
gradually reduce the spatial size of feature maps, then representation resolution is raised by an upsampling or deconvolution.
Multi-scale fusion is realised by skip connections. It allows the final predictions been performed on multiple resolution
layers with both semantically strong and location sensitive information. Such a design can be found in both Faster RCNN+ FPN,
MaskRCNN and YOLOv3 (DarkNet53)

## YOLOv3
The backbone in YOLOv3 is DarkNet 53:
[![6ilNvt.png](https://s3.ax1x.com/2021/03/01/6ilNvt.png)](https://imgtu.com/i/6ilNvt)

In the design of Darknet53, it uses stride=2 convolutions 5 times to replace pooling layers to maintian as much information as possible.
[![6iJscd.png](https://s3.ax1x.com/2021/03/01/6iJscd.png)](https://imgtu.com/i/6iJscd)

Moreover, it uses a specially designed residual block multiple times at different stage:
[![6iJI3Q.png](https://s3.ax1x.com/2021/03/01/6iJI3Q.png)](https://imgtu.com/i/6iJI3Q)
It first uses 1*1 convs to reduce the channels in order to reduce the computation cost. Then it increases back the channel 
size by 3*3 convs and use this layer for feature extraction. 
We can also find that such residual block uses few times at shallow layers and many more times at deeper layers:
(1) At shallow layers the feature maps are big, so it will be big cost if uses residual block too often;
(2) Deeper layers contain more semantic information.

The overall structure looks like the following:
[![6iUNxe.png](https://s3.ax1x.com/2021/03/01/6iUNxe.png)](https://imgtu.com/i/6iUNxe)

## YOLOv4
YOLOv4 used the most advanced network architectures such as Cross-Stage-Partial-connections for its advantage on reduced
computation cost without sacrificing accuracy and introduced many new features into the system. 

[![6iULM4.png](https://s3.ax1x.com/2021/03/01/6iULM4.png)](https://imgtu.com/i/6iULM4)

It also introduces the following new features:
[1] mosaic augmentation;
[2] on the backbone, there are new feature like CSPDarknet53、Mish、Dropblock;
[3] Neck uses SPP、FPN+PAN structure;
[4] In loss calculation it uses CIOU_Loss、DIOU_nms;

## YOLOv5
Compared with YOLOv4, YOLOv5 has its own new characristics and the author provides 4 versions for different requirements.
[![6idz8K.png](https://s3.ax1x.com/2021/03/01/6idz8K.png)](https://imgtu.com/i/6idz8K)

#### -Backbone:
[1] In uses CSP block not only in the backbone like YOLOv4, it also uses in the neck section;
[2] It uses "Focus" structure which enables the down sampling without too much information loss;

#### -Neck:
FPN+PAN
[![6i0GeH.png](https://s3.ax1x.com/2021/03/01/6i0GeH.png)](https://imgtu.com/i/6i0GeH)

#### Loss:
GIOU_Loss for bounding box loss, while YOLOv4 uses CIOU_Loss.
Yolov4 uses DIOU_Loss with DIOU_nms, while Yolov5 uses weighted nms.

#### Its 4 structures:
[![6i00l8.png](https://s3.ax1x.com/2021/03/01/6i00l8.png)](https://imgtu.com/i/6i00l8)
It basically uses different width and depth for 4 different versions without much changes. For instance in yolov5s CSP it 
uses 1 res-block :CSP_1, and Yolov5m with CSP_2, Yolov5l with CSP_3, Yolov5x CSP_4 etc.

## YOLOv5 Customized

 In our own application we choose to use YOLOv5x due to its suitable balance between accuracy, inference speed, number of parameters and model size 
for deployment. For our own scenario, we collect 31 short clips (no longer than one minute) that covers 3 camera sites with 
focus on our main site. It has a 30 FPS and 1280 by 720 as well as 704 by 480 two types of resolutions. It also includes 
various lighting conditions for the main site. In order to avoid redundant training images, we select only one image/second,
which in total provides 1233 images and been annotated by 2 annotators. Then we split the dataset into 65% training, 15% 
validation and 20% testing. We use the pretrained weight from MS COCO[] and perform transfer learning on our own dataset 
at batch size of 16 and a total of 80 epochs. It achieves 81.6% precision, 97.8% recall and mAP@.5 at 98.4% in this dataset. 
The speed of inference on average is 2.6ms on single GPU.  

[![6iBitI.png](https://s3.ax1x.com/2021/03/01/6iBitI.png)](https://imgtu.com/i/6iBitI)

For privacy concern, we cannot show the data of our camera footage. The following is the result from a public dataset on YOLOv5:

<iframe width="560" height="315" src="https://youtu.be/AMTgYZu2QQM" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>


# Reference
https://www.programmersought.com/article/29735056140/
https://zhuanlan.zhihu.com/p/143747206
https://zhuanlan.zhihu.com/p/172121380
http://www.julyedu.com/