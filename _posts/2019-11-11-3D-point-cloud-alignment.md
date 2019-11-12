---
layout:     post
title:      3D point cloud alignment
subtitle:   tests and discussion
date:       2019-11-11
author:     Mary Li
header-img: img/2019-10-10-bg.png
catalog: true
tags:
    - 3D mapping and reconstruction
    - point cloud processing
---

_This post intends to discuss some of the 3D point cloud alignment techniques based on data provided by [1]_

## Alignment Techniques
### ICP
ICP is a commonly used technique for precise alignment, where a rigid transformation and rotation is estimated by iterative least square process.
In the following example, two identical 3D point clouds are aligned by different number of iterations:
![MljJln.png](https://s2.ax1x.com/2019/11/12/MljJln.png)
[![MljMTS.md.png](https://s2.ax1x.com/2019/11/12/MljMTS.md.png)](https://imgchr.com/i/MljMTS)

With more iterations:
![Mlj0kF.png](https://s2.ax1x.com/2019/11/12/Mlj0kF.png)
[![Mlj661.md.png](https://s2.ax1x.com/2019/11/12/Mlj661.md.png)](https://imgchr.com/i/Mlj661)

It is noted that in order for this method to converge, it requires a good initial value. So it normally performs after a more gross align been done.alignment


