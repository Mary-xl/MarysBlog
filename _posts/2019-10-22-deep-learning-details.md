---
layout:     post
title:      Deep Learning Details
subtitle:   Initialization,
date:       2019-10-15
author:     Mary Li
header-img: img/2019-10-15-bg.jpeg
catalog: true
tags: 
    -  deep learning
---
_This article tries to summarise some of the detailed specifications of using deep learning neural networks. Some parts may not be accurate and will be modified 
in latter time._

## Initialization
The basic idea is that we want the networks to begin with a smooth forward and backward propagation. If chosen too big, it will lead to gradient explosion. 
If too small, the gradient will vanish. <br>
Classical initialization methods include: Gaussian, Xavier (2010), and Kaiming. <br>

(1) Gaussian: <br>
          mean=0 <br>
          std=1 <br>
By using a all 0 or Gaussian distribution to initialize weight, the problem of gradient explosion and vanish occurs. <br>
![KNkNJf.png](https://s2.ax1x.com/2019/10/24/KNkNJf.png) [1]
(2) A wise choice of initilization method actually depends on the activation function used in the layer.
 Xavier (2010): at a time without activation or tanh <br>
                mean=0 <br>
                std: [![KNkUW8.png](https://s2.ax1x.com/2019/10/24/KNkUW8.png)](https://imgchr.com/i/KNkUW8) [2]
                  
 Kaiming: <br>
          mean=0
          std: [![KNkwQg.png](https://s2.ax1x.com/2019/10/24/KNkwQg.png)](https://imgchr.com/i/KNkwQg) [2]
  
 

## Reference
-[1] https://intoli.com/blog/neural-network-initialization/
-[2] https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94

          
          