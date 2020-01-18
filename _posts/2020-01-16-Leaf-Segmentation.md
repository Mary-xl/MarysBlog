---
layout:     post
title:      "Leaf Segmentation based on Mask RCNN"
subtitle:   
date:       2020-01-16
author:     Mary Li
header-img: img/2019-10-15-bg.jpeg
catalog: true
tags: 
    -  deep learning
---

## Dataset
The dataset was composed of 10,000 synthetically generated Arabidopsis plant images obtained from 
https://research.csiro.au/robotics/our-work/databases/synthetic-arabidopsis-dataset/. Each image (500x500) has
a png color mask (500x500):
![1pPhZT.png](https://s2.ax1x.com/2020/01/18/1pPhZT.png)
![1pPjeK.png](https://s2.ax1x.com/2020/01/18/1pPjeK.png)

I first convert each color mask into gray image and extract single mask for each leaf based the unique
value of the gray scale:
![1pitkF.png](https://s2.ax1x.com/2020/01/18/1pitkF.png)

Then the dataset was split into 0.8:0.2 for training and validation. The directory organization follows 
the same pattern as data-science-bowl-2018 used in https://github.com/matterport/Mask_RCNN,

## Training
The model was initialised with weights trained on the COCO dataset. The backbone is  ResNet101 with FPN, 
which is able to capture objects of various scales via a top-down pathway and lateral connections. 
Because of this structure, we need to set up 5 anchor sizes. The original implementation uses 32 as the 
smallest anchor size, but it was found that it's not performing well on small leaves. I therefore change it to 8 for P2,
which is a biggest feature map layer, 16 for P3, 32 for P4, 64 for P5, 128 for P6.<br>
![1pk5o8.png](https://s2.ax1x.com/2020/01/18/1pk5o8.png)

Question:The image was resized to 512x512 for training since it uses a ResNet as backbone and its size has to be integerx64.
On P2 level, the stride is 4, then the feature map was 128x128.  Each point at the feature map covers 4x4 area in the original
image, but since it has a 8 pixel anchor size, it is true that it covers a 8x8 anchor area with 3 different aspect ratios? <br>

The training is actually a two step process: first  I trained 9 epoches on the heads layers, 
without resnet backbone (including the top-down path of FPN, RPN, and the heads for classification, bbox reggression
and mask generation). And another 70 epoches on “all layers”. The reason for this is because the backbone has been 
pre-trained on COCO, but the heads parameters are randomly initialised. Such training step can be beneficial to the 
network to converge. Sometimes if we only have a small dataset, which we usually do in plant science study, we 
can also freeze the pre-trained backbone and only train the heads.
![1pAjcd.png](https://s2.ax1x.com/2020/01/18/1pAjcd.png)

During the training of the second stage (all layers), the optimization process encountered two plateaus:<br>
One from around epoch 23 to epoch 40<br>
[![1pEdUK.md.png](https://s2.ax1x.com/2020/01/18/1pEdUK.md.png)](https://imgchr.com/i/1pEdUK)
[![1pEBCD.md.png](https://s2.ax1x.com/2020/01/18/1pEBCD.md.png)](https://imgchr.com/i/1pEBCD)

I first change the learning rate from  LEARNING_RATE = 0.001 to 0.0001, but the loss didn't go down (epoch 30 to 40). Then I changed 
the opimizer from SGD to Adagrad and it successfully enables the optimizer step over the plateau.
![1pEHrn.png](https://s2.ax1x.com/2020/01/18/1pEHrn.png)

Question: However, around apoch 60 to 70, the loss stops going down again. What strategies should I take? Why previously I changed
from SGD to Adagrad and it helps? <br>




 
