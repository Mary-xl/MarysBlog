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

### Literature for VOT

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

### Literature for MOT
[![6ke0pt.png](https://s3.ax1x.com/2021/03/02/6ke0pt.png)](https://imgtu.com/i/6ke0pt)

### Detection-Based Tracking
Detection-Based Tracking: objects are first detected and then linked into trajectories. This strategy is also commonly 
referred to as “tracking-by-detection”. Given a sequence, type-specific object detection or motion detection (based on 
background modeling) is applied in each frame to obtain object hypotheses,then (sequential or batch) tracking is conducted 
to link detection hypotheses into trajectories. There are two issues worth noting. First, since the object detector is trained 
in advance, the majority of DBT focuses on specific kinds of targets, such as pedestrians, vehicles or faces. Second, the 
performance of DBT highly depends on the performance of the employed object detector.

Such detection based tracking pipeline normally consists of 3 modules: detector, tracking and detection-to-tracker association. 
[![6knwsf.png](https://s3.ax1x.com/2021/03/02/6knwsf.png)](https://imgtu.com/i/6knwsf)
from https://mp.weixin.qq.com/s/BywGtfmcY7sinERWLxZdjQ

#### -Detector
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
analysis seems to be a good choice. But it also depends on the size of the objects. The most recent development on the YOLO
series make it the best choice for on-line object detection and tracking. There are reviews on YOLO series in my blog.

#### -Tracking 
The basic idea for tracking is locating an object in successive frames of a video. For online tracking it’s finding an 
object in the current frame given we have tracked the object successfully in all ( or nearly all ) previous frames. 
In some literature this stage is referred to as “motion prediction stage”. One approach for tracking is to build a motion model, 
where the object location of future frames can be predicted based on knowledge of current and previous frames , including 
locations and velocity. One method is to treat trajectory as state-space models like Kalman or Particle Filters.

#### -Detection-to-tracker Association

Each target’s bounding box geometry is estimated by predicting its new location in the current frame based on previous frame 
and Kalman Filter. Then the association problem is to assign the newly detected bboxes to these predicted locations for tracked targets. 

In many approaches, the assignment cost matrix is computed as the intersection-over-union (IOU) distance between each detection 
and all predicted bounding boxes from the existing targets. The assignment is solved optimally using the Hungarian algorithm. 
Additionally, a minimum IOU is imposed to reject assignments where the detection to target overlap is less than IOU min (eg. SORT).

 The most important and challenging part of such an approach is the data association between detection and tracked objects.
 SORT, one of the simplest and efficient on-line MOT algorithm, associate detected object across frames by maximizing the 
 IOU between neighbouring frames. Such an approach however fails in many real-world scenarios where pure IOU may increase 
 chances of ID switch for non-rigid objects or occlusion happens. DeepSORT, which improves SORT by introducing deep learning
 based appearance feature into MOT, is a classical algorithm that has been widely adopted by both academic community and industry.
 It not only combines motion metrics with representational power of deep networks for data association, more importantly,
 it breaks the limit of only considering similarities between neighbouring frames by Cascade Matching over longer sequences.
 By considering both short term and long term occlusions in the data association process, the algorithm gets a more robust 
 tracking performance than its counterparts for online tracking. 

### DeepSORT
[![6knlqO.png](https://s3.ax1x.com/2021/03/02/6knlqO.png)](https://imgtu.com/i/6knlqO)
from https://augmentedstartups.medium.com/deepsort-deep-learning-applied-to-object-tracking-924f59f99104
[![6kncJs.png](https://s3.ax1x.com/2021/03/02/6kncJs.png)](https://imgtu.com/i/6kncJs)
from https://mp.weixin.qq.com/s/BywGtfmcY7sinERWLxZdjQ

Due to privacy concern, I didn't include the pedestrian tracking and counting demo, instead, I reconfigure the model to make
it a small apple tracking and counting demo:

-The tracking algorithm is depended on Deep SORT;
-The target is currently changed to "apple" since this class has been included in the COCO dataset, commonly seen on conveyor 
belt application and pre-trained on Yolo-v3. The detector can be further finetuned for new dataset;
-A counting mechanism has been added to the system including counting of both directions;

<iframe width="560" height="315" src="https://youtu.be/AShXxvh9Dps" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
<div class="youtube-player" data-id="AShXxvh9Dps"></div>