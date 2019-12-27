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

## 1. Excerpt

### 1.1 RPN 
[![lEq12D.md.png](https://s2.ax1x.com/2019/12/27/lEq12D.md.png)](https://imgchr.com/i/lEq12D)
<center> [1] </center> <br>

In the abstract, the author points out two key points: <br>
(1) "We further merge RPN and Fast R-CNN into a single network by sharing their convolutional features"
The region proposal process in Faster R-CNN using RPN depends on the feature map extracted by the CNN backbone, and this
feature map is also used in the following Fast R-CNN detection.
![lEXlGV.png](https://s2.ax1x.com/2019/12/27/lEXlGV.png)
<center> [2] </center> <br>

(2) In the training process, both RPN and Fast R-CNN generate loss. These two sets of loss (2 stages) alternate training the
backbone CNN until the added loss from the two subnetwork smaller than certain threshold.
![lELloq.png](https://s2.ax1x.com/2019/12/27/lELloq.png)
<center> [3] </center> <br>

### 1.2 anchor
[![lVnt1O.md.png](https://s2.ax1x.com/2019/12/27/lVnt1O.md.png)](https://imgchr.com/i/lVnt1O)
 ![lVn8tx.png](https://s2.ax1x.com/2019/12/27/lVn8tx.png)
 <center> [1] </center> <br>
 
 
 
## Reference
[2]https://au.mathworks.com/help/vision/ug/getting-started-with-r-cnn-fast-r-cnn-and-faster-r-cnn.html <br>