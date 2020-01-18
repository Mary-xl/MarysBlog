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

## 1. Dataset
The dataset was composed of 10,000 synthetically generated Arabidopsis plant images. Each image (500x500) has
a png color mask (550x550):
![1pPhZT.png](https://s2.ax1x.com/2020/01/18/1pPhZT.png)
![1pPjeK.png](https://s2.ax1x.com/2020/01/18/1pPjeK.png)

I first convert each color mask into gray image and extract single mask for each leaf based the unique
value of the gray scale:
![1pitkF.png](https://s2.ax1x.com/2020/01/18/1pitkF.png)

Then the dataset was split into 0.8:0.2 for training and validation. The directory organization follows 
the same pattern as data-science-bowl-2018 used in https://github.com/matterport/Mask_RCNN,

## 2. Training
The model was initialised with weights trained on the COCO dataset. The backbone is  ResNet101 with FPN, 
which is able to capture objects of various scales via a top-down pathway and lateral connections. 
Because of this structure, we need to set up 5 anchor sizes. The original implementation uses 32 as the 
smallest anchor size, but it was found that it's not performing well on small leaves. I therefore change it to 8 for P2,
which is a biggest feature map layer, 16 for P3, 32 for P4, 64 for P5, 128 for P6.<br>
![1pk5o8.png](https://s2.ax1x.com/2020/01/18/1pk5o8.png)
see more details in discussion "RPN Targets" section. <br>

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

**Question: However, around apoch 60 to 70, the loss stops going down again. What strategies should I take? Why previously I changed
from SGD to Adagrad and it helps? <br>**

## 3. Result and discussion
Based on the current model (not completed converged though), I did some evaluation:

[![1pVUMj.md.png](https://s2.ax1x.com/2020/01/18/1pVUMj.md.png)](https://imgchr.com/i/1pVUMj)

[![1pVass.md.png](https://s2.ax1x.com/2020/01/18/1pVass.md.png)](https://imgchr.com/i/1pVass)

[![1pVyJU.md.png](https://s2.ax1x.com/2020/01/18/1pVyJU.md.png)](https://imgchr.com/i/1pVyJU)

[![1pV6WF.md.png](https://s2.ax1x.com/2020/01/18/1pV6WF.md.png)](https://imgchr.com/i/1pV6WF)


### 3.1 Backbone and feature extraction
In the following I will generate more detailed information to inspect the training process based on https://github.com/matterport/Mask_RCNN:

#### ResNet with FPN
"The original implementation of Faster R-CNN with ResNets extracted features from the final convolutional layer of the 4-th stage, which we call C4. 
This backbone with ResNet-50, for example, is denoted by ResNet-50-C4. This is a common choice.  
We also explore another more effective backbone recently proposed by Lin et al., called a Feature Pyramid Network (FPN). 
FPN uses a top-down architecture with lateral connections to build an in-network feature pyramid from a single-scale input. 
Faster R-CNN with an FPN backbone extracts RoI features from different levels of the feature pyramid according to their scale, 
but otherwise the rest of the approach is similar to vanilla ResNet. Using a ResNet-FPN backbone for feature extraction with Mask R-CNN gives excellent
gains in both accuracy and speed."[1]
<br>
"The goal of this paper is to naturally leverage the pyramidal shape of a ConvNet’s feature hierarchy while creating a feature pyramid that has strong 
semantics at all scales.   Using FPN in a basic Faster R-CNN system, our method achieves state-of-the-art single-model results on the COCO detection 
benchmark without bells and whistles, surpassing all existing single-model entries including those from the COCO 2016 challenge winners. In addition, 
our method can run at 6 FPS on a GPU and thus is a practical and accurate solution to multi-scale object detection.  "[1]
<br>
#### About ResNet
 * Skip connection solves the gradient vanish problem <br>
 * Skip connection solves 
 * The author adopt batch normalization (BN) right after each convolution and before activation, initialize the weights as in and train all plain/residual nets from scratch.They do not use dropout .  


### 3.2 Stage 1: Region Proposal Network
#### RPN Targets 
"We adapt RPN by replacing the single-scale feature map with our FPN. We attach a head of the same design (3×3 conv and two sibling 1×1 convs) to
each level on our feature pyramid. Because the head slides densely over all locations in all pyramid levels, it is not necessary to have multi-scale anchors
on a specific level.Instead, we <font color=Coral>assign anchors of a single scale to each level. Formally, we define the anchors to have areas of {$32^2$ , $64^2$ , $128^2$ , $256^2$ , $512^2$ } pixels
on {P2, P3, P4, P5, P6} respectively</font>. We also use anchors of multiple aspect ratios {1:2, 1:1, 2:1} at each level. So in total there are 15 sizes anchors over the pyramid."[1] 

Noted that in this leaf segmentation case, the anchor size for the 5 levels are chosen as [8,16,32,64,128]: <br>
The total anchors' number per image is:<br><br>

P2:512/4(stride)=128 size, anchor_num=128*128*3,  the chosen anchor (1x1 ratio) covers 8x8 pixels in the original image<br>
P3:512/8=64,anchor_num=64*64*3, the chosen anchor covers 16*16 pixels in the original image  <br>
P4:512/16=32, anchor_num=32*32*3, the chosen anchor covers 32*32 pixels in the original image <br>
P5:512/32=16,anchor_num=16*16*3, the chosen anchor covers 64*64 pixels in the original image <br>
P6:512/32=8, anchor_num=8*8*3, the chosen anchor covers 128*128 pixels in the original image <br> 
Sum(P2+P3+P4+P5+P6)=65472<br> <br>

In the training process, RPN targets are generated in each batch, with 256 samples/batch and IoU>0.7 positive, IoU<0.3 as negative:
[![1poPXR.md.png](https://s2.ax1x.com/2020/01/18/1poPXR.md.png)](https://imgchr.com/i/1poPXR)

 RPN_TRAIN_ANCHORS_PER_IMAGE = 256 so target_rpn_bbox.shape=(256,4)<br>
 
#### RPN Predictions
[![19pOBT.md.png](https://s2.ax1x.com/2020/01/18/19pOBT.md.png)](https://imgchr.com/i/19pOBT) 

It has 4 steps: <br>
(1) Based on the prediction of confidence scores (objectness score), obtain the top 6000 proposals (anchors);<br>
[![199Y5Q.md.png](https://s2.ax1x.com/2020/01/18/199Y5Q.md.png)](https://imgchr.com/i/199Y5Q)

(2) Using their corresponding bbox offset values to get their adjusted bbox on the original image;<br>
(3) Remove the ones that exceed the image boundary;<br>
[![199yaF.md.png](https://s2.ax1x.com/2020/01/18/199yaF.md.png)](https://imgchr.com/i/199yaF)

(4) Using NMS to get the final 2000 proposals in training/ 1000 proposals in evaluation (roi). <br>
[![199oVO.md.png](https://s2.ax1x.com/2020/01/18/199oVO.md.png)](https://imgchr.com/i/199oVO)
 
