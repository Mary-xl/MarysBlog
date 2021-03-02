---
layout:     post
title:      "Object Tracking"
date:       2020-10-01
author:     Mary Li
header-img:
catalog: true
tags:
- deep learning
- tracking
---
_In 2020, I started a new project on video surveillance, where deep learning based techniques like object detection, 
multiple object tracking, human action recognition are heavily used and modified for our own application._

# Single object tracking (VOT)
[![6kETne.png](https://s3.ax1x.com/2021/03/02/6kETne.png)](https://imgtu.com/i/6kETne)

[![6kELtI.png](https://s3.ax1x.com/2021/03/02/6kELtI.png)](https://imgtu.com/i/6kELtI)

# Multiple object tracking (MOT)
Although it seems multiple object object tracking only increases the number of objects for tracking, 
it actually has quite different objectives to achieve and uses quite different methodologies. Some major differences are :
- VOT deals with a single predefined target in the scene, which usually stays in the scene the whole tracking period;
- MOT needs to detect multiple targets, which usually belongs to certain type, and track their movement. During the time 
  the targets can appear,move, been occluded,  and leave the scene;

Therefore VOT usually focus on target re-localisation, while MOT focus on the data association of detected targets.

MOT has two main stream applications: detection-based tracking and detection-free tracking. While the latter one can be 
seen as multiple VOT, the first one attracts more interest and has a wider range of applications. 

Detection-Based Tracking: objects are first detected and then linked into trajectories. This strategy is also commonly 
referred to as “tracking-by-detection”. Given a sequence, type-specific object detection or motion detection (based on 
background modeling) is applied in each frame to obtain object hypotheses,then (sequential or batch) tracking is conducted 
to link detection hypotheses into trajectories. There are two issues worth noting. First, since the object detector is trained 
in advance, the majority of DBT focuses on specific kinds of targets, such as pedestrians, vehicles or faces. Second, the 
performance of DBT highly depends on the performance of the employed object detector.

Such detection based tracking pipeline normally consists of 3 modules: detector, tracking and detection-to-tracker association. 

### -Detector
 Target objects are detected on each frame and output in the form of a bbox. Detectors are always trained in advance. Common pedestrian detector are:

- Non-deep learning detectors
      These methods use manual features for visual representation of the target object.
      -Dense Optical Flow: dense optical flow estimates the motion vector of every pixel.
     -Sparse Optical Flow: first locates a few feature points on the frame then estimates the motion vector for these feature points. Lucas‐Kanade Flow.  It assumes small motions. 
     Some challenges with optical flow in tracking: computationally expensive for dense optical flow; for sparse optical flow, ambiguity of feature selection, adding, deletion, as well as handling of occlusion and drifting errors (no correspondence constraint).  It is fundamentally down to the local feature level instead of tracking of an object. So if optical flow been used, a more sophisticated model needs to be built on top.  
      -Color histogram
      -HOG (2005) +image pyramid+sliding window+ SVM classifer 

    There are many other features and ideas been used for tracking, such as MeanShift etc. The main problem for these methods are the handling of occlusion and deformation. 
          
- Deep learning based approaches
    With the booming of deep learning, more accurate and comprehensive convolutional neural network based classifier& detectors have been developed. Some popular object detection frameworks, such as Faster RCNN, SSD and YOLO: 

      - Faster RCNN 
      - SSD 
      - YOLO series 

[![6kZrdJ.png](https://s3.ax1x.com/2021/03/02/6kZrdJ.png)](https://imgtu.com/i/6kZrdJ)
Generally, Faster RCNN achieves the best accuracy but a low speed. YOLO performs very well on speed with some sacrifice 
on accuracy (especially for small objects). SSD has a balanced trade-off between accuracy and speed. Using SSD for video
analysis seems to be a good choice. But it also depends on the size of the objects. 


