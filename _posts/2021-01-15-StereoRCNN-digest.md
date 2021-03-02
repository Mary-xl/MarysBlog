---
layout:     post
title:      "StereoRCNN Digest"
date:       2021-01-15
author:     Mary Li
header-img:
catalog: true
tags:
- deep learning
- object detection
- 3D detection
---
_This is a digest article on StereoRCNN_

This method does not require depth input and 3D position, but it performs better than traditional stereo based methods.
On Kitti, for instance, the 3D detection AP outperforms 30% AP.

## Overall architecture
[![6Fm7Sx.png](https://s3.ax1x.com/2021/03/02/6Fm7Sx.png)](https://imgtu.com/i/6Fm7Sx)

As can be observed, it is an extension over Faster-RCNN, from 2d bbox prediction to 3D bbox prediction. The overall network
also consists of 3 parts: backbone, RPN, and regression heads.

### Backbone
The backbone uses ResNet 101 + FPN, which is identical to the backbone used in Mask RCNN. With a top-down pathway, the smalled
but most semantically strong feature map will be up-sampled to different scales and features from to bottom-up pathway will be 
fused with it to enable such feature map been both sematically strong and location sensitive for certain scales. Such a structure
can enable objects of different sizes/scales been detected accurately.

### Stereo RPN
The same way as traditional RPN in Mask RCNN, interest areas are classified into 2 groups (fore/back ground) and regressed 
based on 3 anchors of each location on 5 different scale feature maps. Based on this, stereo RPN process regress on both left
and right images and get output like [u,v,w,h,u',v',w',h']. Since the stereo pair have been rectified, v'= v and h'=h, leaving
[[u,w,u’,w’,v,h]]
[![6Fyt1I.png](https://s3.ax1x.com/2021/03/02/6Fyt1I.png)](https://imgtu.com/i/6Fyt1I)

### Stereo R-CNN: stereo regression and keypoint prediction
The key innovation of this paper is in this part and the following ones. After Stereo RPN, ROI Align is applied on both sides. 
[![6F6ORs.png](https://s3.ax1x.com/2021/03/02/6F6ORs.png)](https://imgtu.com/i/6F6ORs)

#### -stereo regression
Four branches are doing the following:
(1) 2D bounding box regression
(2) classification
(3) dimension: W,H,L of the 3D bounding box
(4) viewpoint [sin, cos]

#### -keypoint prediction
It predicts one of the four corners of the 3D bounding box on the bottom. It only predicts one is because when project to
2D image, there will be only one point inside the 2D bounding box (cropped region after ROI process)
[![6FcT61.png](https://s3.ax1x.com/2021/03/02/6FcT61.png)](https://imgtu.com/i/6FcT61)

### 3D box estimation
After obtaining the information on 2D bounding boxes, dimensions, and viewing angles, 3D bounding box can be calculated
through a projection equation:

[![6F2uPe.png](https://s3.ax1x.com/2021/03/02/6F2uPe.png)](https://imgtu.com/i/6F2uPe)
[![6kkLbd.png](https://s3.ax1x.com/2021/03/02/6kkLbd.png)](https://imgtu.com/i/6kkLbd)

Once the projection relationship can be built for each point, and K is obtained in the calibration process, the (x,y,z)
which is the 3D center location of the 3D bounding box can be calculated by solving an optimization problem using Gauss-Newton.
[![6kAmGV.png](https://s3.ax1x.com/2021/03/02/6kAmGV.png)](https://imgtu.com/i/6kAmGV)

### Dense 3D box alignment
[![6FR8SJ.png](https://s3.ax1x.com/2021/03/02/6FR8SJ.png)](https://imgtu.com/i/6FR8SJ)
It basically uses the calculated depth value (box center location relative to the camera) b to get disparity. And uses the
disparity to get corresponding point on the right image. Then compare the value (pixel value) on the corresponding locations 
of the stereo pair.
