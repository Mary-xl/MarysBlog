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



