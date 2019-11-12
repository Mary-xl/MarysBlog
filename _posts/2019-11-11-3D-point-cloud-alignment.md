---
layout:     post
title:      3D point cloud processing
subtitle:   tests and discussion
date:       2019-11-11
author:     Mary Li
header-img: img/2019-10-10-bg.png
catalog: true
tags:
    - 3D mapping and reconstruction
    - point cloud processing
---

_This post intends to discuss some of the commonly used 3D point cloud processing techniques based on data provided by [1] and the PCL library tutorial [2]_

## Alignment Techniques
### ICP
ICP is a commonly used technique for precise alignment, where a rigid transformation and rotation is estimated by iterative least square process.
In the following example, two identical 3D point clouds are aligned by different number of iterations:
![MljJln.png](https://s2.ax1x.com/2019/11/12/MljJln.png)
[![MljMTS.md.png](https://s2.ax1x.com/2019/11/12/MljMTS.md.png)](https://imgchr.com/i/MljMTS)
Noted that the transformation and rotation between two 3D point clouds can be discribed by a 3*3 rotation (3 DoF) abd 3*1 translation (3 DoF) matrices, or a
4*4 matrix with 6 DoF like:

With more iterations:
![Mlj0kF.png](https://s2.ax1x.com/2019/11/12/Mlj0kF.png)
[![Mlj661.md.png](https://s2.ax1x.com/2019/11/12/Mlj661.md.png)](https://imgchr.com/i/Mlj661)

It is noted that in order for this method to converge, it requires a good initial value. So it normally performs after a more gross align has been done.

###  Normal Distributions Transform (NDT)
NDT can be used for the alignment of big point clouds based on the statistical probabilities of point cloud registration.
A good example can be found from: http://pointclouds.org/documentation/tutorials/normal_distributions_transform.php
The two point clouds:
![M1V8sO.png](https://s2.ax1x.com/2019/11/12/M1V8sO.png)
Aligned point cloud:
[![M1ZFkd.md.png](https://s2.ax1x.com/2019/11/12/M1ZFkd.md.png)](https://imgchr.com/i/M1ZFkd)

It is noted that in this algorithm, several parameters need to be tuned for specific experiments, depending on the actual scale.
![M1e0r8.png](https://s2.ax1x.com/2019/11/12/M1e0r8.png)
<center>[2]</center>

### to be continued...

## Reference
[1] http://www.pclcn.org/product/showproduct.php?lang=cn&id=20 <br>
[2] http://pointclouds.org/documentation/tutorials/ <br>
