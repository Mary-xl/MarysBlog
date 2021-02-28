---
layout:     post
title:      "Object detection: from YOLOv3 to YOLOv5 customized -1"
date:       2020-05-05
author:     Mary Li
header-img:
catalog: true
tags:
- deep learning
- object detection
---
_In 2020, I started a new project on video surveillance, where deep learning based techniques like object detection, 
multiple object tracking, human action recognition are heavily used and modified for our own application._


## Introduction

Object detection has been one of the biggest area that boomed by the deep learning technology. In this article, I will 
divide the development into two parts: backbone and overall structure . More and more powerful backbone will extract 
both semantically strong and location sensitive features for object detections, coupled with carefully designed architecture,
object detection is heading to a direction that is both accurate and fast.

## Backbone

[![696526.png](https://s3.ax1x.com/2021/02/28/696526.png)](https://imgtu.com/i/696526)

A general and basic backbone looks like above, which consists of normal conv layers followed by pooling layers and final
fully connected layers. Theoretically, a deeper and wider network is often more powerful in feature extraction. 

VGG, for instance uses small kernel to replace large kernels (same perception field with less parameters
and more semantically strong features) and result in a deep network.  GoogleNet on the other hand uses different kernel sizes and a wider network to 
improve the overall improvement. However, a deeper network usually leads to gradient vanishing problem, which prohibits 
the development of a deeper neural network.

[![69RgQx.png](https://s3.ax1x.com/2021/02/28/69RgQx.png)](https://imgtu.com/i/69RgQx)

Until the invention of ResNet, such problem has been solved by skip connections in these so called "residual layers". Moreover,
it facilitate the feature propagation from shallow layers to deep layers, which makes deep neural networks been "deep" without obstacle.

[![69flCj.png](https://s3.ax1x.com/2021/02/28/69flCj.png)](https://imgtu.com/i/69flCj)

Following this trend, more and more new powerful networks have been invented.

"Recent work has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if
they contain shorter connections between layers close to the input and those close to the output. In this paper, we embrace 
this observation and introduce the Dense Convolutional Network (DenseNet)". In DenseNet, for each layer, the feature-maps 
of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. It brings 
the following advantages:
(1) mitigate feature vanishing problem;
(2) strengthen feature propagation;
(3) encourage feature reuse;
(4) decrease number of parameters.

