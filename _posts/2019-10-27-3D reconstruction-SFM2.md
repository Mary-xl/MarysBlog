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

In previous article, the traditional SFM and its technical details are summarised and discussed. This article intends to introduce some more recent development in this field, including techniques used in different phase of SFM.
Meanwhile, some of its application in large scale 3D reconstruction is also discussed. I will include its application in SLAM in the future. 


### 1. Feature extraction and Image Retrival
In SFM, the first phase is feature extraction and matching. Traditional SFM used manual-feature such as harris corners and SIFT. Recent improvement in CNN enables many learning-based features been developed, such as LIFT[1], and GeoDesc[2] etc. The former one was developed
for general CV applications, while the latter one was learning specifically from SFM results so therefore shows good results when used for 3D reconstruction applications. It actually outperforms SIFT as the author claimed:
[![KRoBjO.md.png](https://s2.ax1x.com/2019/10/29/KRoBjO.md.png)](https://imgchr.com/i/KRoBjO)
<center>[2]</center>

One major difference between small scene SFM, SLAM and large scale 3D reconstruction is they have different scenarios for image matching:
(1) Small scene project with limited number of images can do pair-wise matching easily. <br>
(2) SLAM have sequential images, which therefore makes it easy to find relevant images for matching. <br>
(3) For large scale 3D reconstruction, especially ones with random images (not sequential or ordered), pair-wise matching (O(n^2)) is not realistic. <br>

So therefore there is a pre-processing step to find relevant images (images with the same objects) before actual SFM takes place. A technique named "vocabulary tree", or "Bag-of-words" has been used. In my paper (<a href="http://marysfishingpool.tk///2019/10/07/Vision-based-positioning/"> Vision-based-positioning </a>) where images of the same building needs to be identified from the image database, 
a similar one  "feature-based voting" is used. Such a technique has been widely used in object detection before the CNN era. 

### 2. Camera registration
As mentioned in previous article, camera registration is usually performed incrementally. It provides a robust way of adding new cameras into the system as it repeatedly perform bundle adjustment. It however takes long time due to this reason.
Some important work are from (<a href="http://phototour.cs.washington.edu/ /"> photo tourism </a>) [3] where an initial pair was selected and the following cameras were registered depend on the number of common
points("tracks") with the available registered cameras. Altough the group modified the research in 2008 to replace the initial pair to a minimum set of cameras representing the whole scene, it is essentially an incremental registration approach.
<br>
[![KWkWjO.md.png](https://s2.ax1x.com/2019/10/29/KWkWjO.md.png)](https://imgchr.com/i/KWkWjO)
<center>[3]</center>







## Reference
[1] Yi, K.M., Trulls, E., Lepetit, V., & Fua, P. (2016). LIFT: Learned Invariant Feature Transform. ECCV.<br>
[2] "GeoDesc: Learning Local Descriptors by Integrating Geometry Constraints", Zixin Luo, Tianwei Shen, Lei Zhou, Siyu Zhu, Runze Zhang, Yao Yao, Tian Fang and Long Quan. <br>
[3] Snavely, N., Seitz, S. M. & Szeliski, R. (2006). Photo tourism: Exploring photo collections in 3D. SIGGRAPH Conference Proceedings (p./pp. 835--846), New York, NY, USA: ACM Press. ISBN: 1-59593-364-6 <br>