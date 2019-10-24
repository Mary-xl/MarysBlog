---
layout:     post
title:      Deep Learning Details
subtitle:   Pre-processing (PCA), Initialization,
date:       2019-10-15
author:     Mary Li
header-img: img/2019-10-15-bg.jpeg
catalog: true
tags: 
    -  deep learning
---
_This article tries to summarise some of the detailed specifications of using deep learning neural networks. Some parts may not be accurate and will be modified 
at latter time._

## 1. Data pre-processing
![KN7ytI.png](https://s2.ax1x.com/2019/10/24/KN7ytI.png) <center> [4] </center>

Normally we do a data normalization followed by a dimension reduction n->k (PCA). 
The key idea is to reduce the size of data while maintaining as high variance (信息量）as possible. <br>

(1) Given m samples of training data X
[![KN4X8O.png](https://s2.ax1x.com/2019/10/24/KN4X8O.png)](https://imgchr.com/i/KN4X8O) <center> [1] </center>
if features of a sample are of different scale, do xi=xi/std. <br>

(2) Get covariance of the X: <br>
[![KN5fot.png](https://s2.ax1x.com/2019/10/24/KN5fot.png)](https://imgchr.com/i/KN5fot) 

(3) Use SVD to find the eigenvectors and choose the first k as new axes: <br> 
[![KNInSO.png](https://s2.ax1x.com/2019/10/24/KNInSO.png)](https://imgchr.com/i/KNInSO) <center> [1] </center>
![KNI3TI.png](https://s2.ax1x.com/2019/10/24/KNI3TI.png) <center> [1] </center>
For these n columns of eigenvectors, get the first k columns as the dimension you wish to reduce to. 
![KNI7h6.png](https://s2.ax1x.com/2019/10/24/KNI7h6.png) <center> [1] </center>

(4) Project the original n dimension data onto these k dimension sub-space to get reduced data:
![KNosDH.png](https://s2.ax1x.com/2019/10/24/KNosDH.png)


sample code : <br>
![KN7xHJ.png](https://s2.ax1x.com/2019/10/24/KN7xHJ.png)


## 2. Initialization
The basic idea is that we want the networks to begin with a smooth forward and backward propagation. If chosen too big, it will lead to gradient explosion. 
If too small, the gradient will vanish. <br>
Classical initialization methods include: Gaussian, Xavier (2010), and Kaiming. <br>

### Using random or Gaussian initialization
Gaussian: <br>
          mean=0 <br>
          std=1 <br>
By using a all 0 or Gaussian distribution to initialize weight, the problem of gradient explosion and vanish occurs. <br>
![KNkNJf.png](https://s2.ax1x.com/2019/10/24/KNkNJf.png) <center> [2] </center><br>

### Using Xavier or Kaiming methods
 A wise choice of initilization method actually depends on the activation function used in the layer.
 Xavier (2010): at a time without activation or using tanh <br>
                mean=0 <br>
                std: [![KNkUW8.png](https://s2.ax1x.com/2019/10/24/KNkUW8.png)](https://imgchr.com/i/KNkUW8) <center> [3] </center>
                  
 Kaiming: <br>
          mean=0
          std: [![KNkwQg.png](https://s2.ax1x.com/2019/10/24/KNkwQg.png)](https://imgchr.com/i/KNkwQg) <center> [3] </center>
  
 

## Reference
-[1] https://www.youtube.com/watch?v=rng04VJxUt4
-[2] https://intoli.com/blog/neural-network-initialization/ <br>
-[3] https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
-[4] "Getting Started with Machine Learning" by Jim Liang

          
          