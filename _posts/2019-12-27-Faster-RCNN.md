---
layout:     post
title:      "Faster R-CNN Digest"
subtitle:   Reading and Digest
date:       2019-12-27
author:     Mary Li
header-img: img/2019-10-15-bg.jpeg
catalog: true
tags: 
    -  deep learning
    -  paper digest 
---

## 1. RPN

### 1.1 RPN benefit
[![lEq12D.md.png](https://s2.ax1x.com/2019/12/27/lEq12D.md.png)](https://imgchr.com/i/lEq12D)
<center> [1] </center> <br>

In the abstract, the author points out: <br>
(1) "We further merge RPN and Fast R-CNN into a single network by sharing their convolutional features"
The region proposal process in Faster R-CNN using RPN depends on the feature map extracted by the CNN backbone, and this
feature map is also used in the following Fast R-CNN detection.
![lEXlGV.png](https://s2.ax1x.com/2019/12/27/lEXlGV.png)
<center> [2] </center> <br>

The benefit of such usage of RPN are:<br>
-- Largely increase the speed (FPS) and reduce computation cost, since Selective Search has been replaced; <br>
-- Increase the accuracy (mAP) of detection, since region proposal was done on feature map (higher level of features) rather than
on the original image. In RCNN for instance, the detection accuracy was based on the performance of selective search on original image.

![lVMs41.png](https://s2.ax1x.com/2019/12/27/lVMs41.png)
<center> [1] </center> <br>
[![lVQ3rD.md.png](https://s2.ax1x.com/2019/12/27/lVQ3rD.md.png)](https://imgchr.com/i/lVQ3rD)

(2) Two losses for RPN region proposal and Fast R-CNN detection: 
![lVnDAI.png](https://s2.ax1x.com/2019/12/27/lVnDAI.png)
<center> [1] </center> <br>
 In the training process, both RPN and Fast R-CNN generate loss. These two sets of loss (2 stages) alternate training the
backbone CNN until the added loss from the two subnetwork smaller than certain threshold.
![lELloq.png](https://s2.ax1x.com/2019/12/27/lELloq.png)
<center> [3] </center> <br>

### 1.2 Anchor and RPN training
 (1) During the training of RPN, the first step is the generation of anchor boxes: 
[![lQtrM8.md.png](https://s2.ax1x.com/2019/12/30/lQtrM8.md.png)](https://imgchr.com/i/lQtrM8)
<center> [5] </center> <br>
 (2) The second step is to  generate the targets of the RPN: pos/neg classification, offset(anchor boxes, GT):
 [![lQNxts.md.png](https://s2.ax1x.com/2019/12/30/lQNxts.md.png)](https://imgchr.com/i/lQNxts)
 <center> [5] </center> <br>
 For pos/neg targets generation:
 [![lQdvAe.md.png](https://s2.ax1x.com/2019/12/30/lQdvAe.md.png)](https://imgchr.com/i/lQdvAe)
 (3) The third step is to use the network to do forward pass, loss with targets, backward to do training on the RPN.
 [![lQRw7R.md.png](https://s2.ax1x.com/2019/12/30/lQRw7R.md.png)](https://imgchr.com/i/lQRw7R)
  <center> [5] </center> <br>
 (4) After training, a size 38*57 feature map can generate 38*57*9=19494 region proposals. Not all these proposals will 
 be forward to ROI pooling. Instead, a NMS is used to filter redundent proposals, retaining only top score proposals with no
 big IoU with other top score proposals.
 
 
 
[![lVnt1O.md.png](https://s2.ax1x.com/2019/12/27/lVnt1O.md.png)](https://imgchr.com/i/lVnt1O)
 ![lVn8tx.png](https://s2.ax1x.com/2019/12/27/lVn8tx.png)
 <center> [1] </center> <br>
 
### 1.3 Use FPN in RPN
In the original paper (baseline) for Faster R-CNN, the RPN uses a single feature map from backbone CNN extraction. 
[![luS6pQ.md.png](https://s2.ax1x.com/2019/12/29/luS6pQ.md.png)](https://imgchr.com/i/luS6pQ)
 <center> [4] </center> <br>
Introducing FPN to RPN will increase the ability of object detection on various scales, since it generates the ROI from multiple layers of 
feature maps:
"Based on the size of the ROI, we select the feature map layer in the most proper scale to extract the feature patches."
[![lupYNT.md.png](https://s2.ax1x.com/2019/12/29/lupYNT.md.png)](https://imgchr.com/i/lupYNT)
 <center> [4] </center> <br>
 
 It is noted the most outstanding improvement was on the AR (average recall: the ability to capture objects):
 [![lupwv9.md.png](https://s2.ax1x.com/2019/12/29/lupwv9.md.png)](https://imgchr.com/i/lupwv9)
  <center> [4] </center> <br>
  
  
## Reference
[2] https://au.mathworks.com/help/vision/ug/getting-started-with-r-cnn-fast-r-cnn-and-faster-r-cnn.html <br>
[4] https://medium.com/@jonathan_hui/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c <br>
[5] http://www.julyedu.com
