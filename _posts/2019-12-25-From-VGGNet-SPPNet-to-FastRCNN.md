---
layout:     post
title:      "From VGGNet, SPPNet to Fast R-CNN"
subtitle:   Reading and Digest
date:       2019-12-25
author:     Mary Li
header-img: img/2019-10-15-bg.jpeg
catalog: true
tags: 
    -  deep learning
    -  paper digest 
---

_This is a paper digest from the classic paper of Fast R-CNN by [1]._

## 1. VGG Backbone
The Fast R-CNN uses a VGG-16 backbone with the following modifications:
(1) It replaces the last max pooling by a ROI pooling;
(2) It replaces the final FC(1000classes FC) to a two branch structure for classification and bbox regression.
[![liFJsO.md.png](https://s2.ax1x.com/2019/12/25/liFJsO.md.png)](https://imgchr.com/i/liFJsO)
 
 ## 2. Why SPP-Net (spatial pyramid pooling)
 ![liFLTJ.png](https://s2.ax1x.com/2019/12/25/liFLTJ.png)
 <center> [2] </center>
 <br>
 In RCNN each region produced by selective search was put into the CNN for feature extraction. Since VGG16 has
 FC layers after conv layers, the author has to make sure that the input regions are adjusted to the same size, thus
 warping/croping the regions. It causes information distortion and loss in the process. SPP-Net is able to tackle this 
 problem by ROI pooling. Any size of the regions can be used as input.
 [![liA3DO.md.png](https://s2.ax1x.com/2019/12/25/liA3DO.md.png)](https://imgchr.com/i/liA3DO)
  <center> [2] </center>
 <br>
 Note that with different sizes of kernel and stride used in the SPP-Net (shown in the above picture, with 4*4+2*2+1*1=21),
 different scale object can be detected.  The kernel and stride sizes can be calculated using: outputS=(inputS-kernel+2padding)/stride+1  
  
  
 ## Reference
 [2] Deepshare.net