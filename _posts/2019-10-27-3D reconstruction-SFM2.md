---
layout:     post
title:      3D Reconstruction- SFM2 
date:       2019-10-27
author:     Mary Li
header-img: img/2019-10-12-bg.png
catalog: true
tags: 
    -  3D mapping and reconstruction
    -  image processing 
---

_In previous article, the traditional (<a href="http://marysfishingpool.tk///2019/10/12/SFM/"> SFM </a>) and its technical details are summarised and discussed. This article intends to introduce SFM from literature point of view, including some more recent development.
Meanwhile, some of its application in large scale 3D reconstruction is also discussed. I plan to include its application in SLAM in the future._


### 1. Feature extraction and Image Retrival
In SFM, the first phase is feature extraction and matching. Traditional SFM used manual-feature such as harris corners and SIFT. Recent improvement in CNN enables many learning-based features been developed, such as LIFT[1], and GeoDesc[2] etc. The former one was developed
for general CV applications, while the latter one was learning specifically from SFM results so therefore shows good results when used for 3D reconstruction applications. It actually outperforms SIFT as the author claimed:
[![KRoBjO.md.png](https://s2.ax1x.com/2019/10/29/KRoBjO.md.png)](https://imgchr.com/i/KRoBjO)
<center>[2]</center>

One major difference between small scene SFM, SLAM and large scale 3D reconstruction is they have different scenarios for image matching:<br>
(1) Small scene project with limited number of images can do pair-wise matching easily. <br>
(2) SLAM have sequential images, which therefore makes it easy to find relevant images for matching. <br>
(3) For large scale 3D reconstruction, especially ones with random images (not sequential or ordered), pair-wise matching (O(n^2)) is not realistic. <br>

So therefore there is a pre-processing step to find relevant images (images with the same objects) before actual SFM takes place. A technique named "vocabulary tree", or "Bag-of-words" has been used. In my paper (<a href="http://marysfishingpool.tk///2019/10/07/Vision-based-positioning/"> Vision-based-positioning </a>) where images of the same building needs to be identified from the image database, 
a similar one  "feature-based voting" is used. Such a technique has been widely used in object detection before the CNN era. 

### 2. Camera registration
As mentioned in previous article, camera registration is usually performed incrementally. It provides a robust way of adding new cameras into the system as it repeatedly perform bundle adjustment. It however takes long time due to this reason.
Some important work are from (<a href="http://phototour.cs.washington.edu/ /"> photo tourism </a>) [3] where an initial pair was selected and the following cameras were registered depend on the number of common
points("tracks") with the available registered cameras. Altough the research group modified the research in 2008 to replace the initial pair to a minimum set of cameras representing the whole scene, it is essentially an incremental registration approach.
Another important work (<a href="http://grail.cs.washington.edu/projects/rome// /"> Building Rome in a Day </a>) [4] introduced a new, parallel distributed matching system that can match massive collections of images very quickly. However, like SLAM,
 if these incremental adjustment approaches do not use global information as final "close loop check", there can be large drifting error.
<br>
[![KWkWjO.md.png](https://s2.ax1x.com/2019/10/29/KWkWjO.md.png)](https://imgchr.com/i/KWkWjO)
<center>[3]</center>

Therefore some researchers adopt "global method" for camera registration. The basic idea is to optimize the whole system based on motion avaraging[9].
_Global SfM is different from incremental SfM in that it considers the entire view graph at the same time instead of incrementally adding more and more images to the Reconstruction. Global SfM methods have been proven to be very fast with comparable or better accuracy to incremental SfM approaches (See [5], [6], [7]), and they are much more readily parallelized[8]._
<br>
The advantage of such method is efficiency and avoiding drifting error due to global information. But it is more sensitive to outliers than incremental SFM. 
(I'm personally not doing research in this direction so I just include some basic information here). <br>
 
### 3. Bundle Adjustment
As mentioned in previous article, bundle adjustment plays an essential role in SFM. It uses reprojection error as loss function to optimize a system parameters consisted of camera poses and 3D points.
 


## Reference
[1] Yi, K.M., Trulls, E., Lepetit, V., & Fua, P. (2016). LIFT: Learned Invariant Feature Transform. ECCV.<br>
[2] "GeoDesc: Learning Local Descriptors by Integrating Geometry Constraints", Zixin Luo, Tianwei Shen, Lei Zhou, Siyu Zhu, Runze Zhang, Yao Yao, Tian Fang and Long Quan. <br>
[3] Snavely, N., Seitz, S. M. & Szeliski, R. (2006). Photo tourism: Exploring photo collections in 3D. SIGGRAPH Conference Proceedings (p./pp. 835--846), New York, NY, USA: ACM Press. ISBN: 1-59593-364-6 <br>
[4] Sameer Agrwal , Yasutaka Furukawa , Noah Snavely , Ian Simon , Brian Curless , Steven M. Seitz , Richard Szeliski, Building Rome in a day, Communications of the ACM, v.54 n.10, October 2011  [doi>10.1145/2001269.2001293]<br>
[5] Jiang, N. and Cui, Z., and Tan, P. A Global Linear Method for Camera Pose Registration, International Conference on Computer Vision (ICCV), 2013. <br>
[6] Moulon, P. and Monasse, P. and Marlet, R. Global Fusion of Relative Motions for Robust, Accurate and Scalable Structure from Motion International Conference on Computer Vision (ICCV), 2013.<br>
[7] Wilson, K. and Snavely, N. Robust Global Translation with 1DSfM European Conference on Computer Vision, 2014.
[8] http://theia-sfm.org/sfm.html
[9] V. M. Govindu. Combining two-view constraints for motionestimation. InCVPR, 2001.