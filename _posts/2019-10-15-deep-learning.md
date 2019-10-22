---
layout:     post
title:      Deep Learning
subtitle:   Reading and Thought
date:       2019-10-15
author:     Mary Li
header-img: img/2019-10-15-bg.jpeg
catalog: true
tags: 
    -  deep learning
    -  paper digest 
---
_From this week, I will begin a paper digest series in which classic papers are read, extracted and digest by asking relevant questions (I could think of), then give reference answers or my thought.
 This article is the digest from [1]_

## 1. Excerpt
_To properly adjust the weight vector, the learning algorithm computes a gradient vector that, for each weight, indicates by what amount
the error would increase or decrease if the weight were increased by a tiny amount. The weight vector is then adjusted in the opposite direction to the gradient vector._
![Screenshot from 2019-10-09 21-58-07.png](https://i.loli.net/2019/10/16/LC3pmPNd6a95yB8.png)[2]

_The objective function, averaged over all the training examples, can be seen as a kind of hilly landscape in the high-dimensional space of
weight values. The negative gradient vector indicates the direction
of steepest descent in this landscape, taking it closer to a minimum,
where the output error is low on average._

_A deep-learning architecture is a multilayer stack of simple mod-
ules, all (or most) of which are subject to learning, and many of which
compute non-linear input–output mappings. Each module in the
stack transforms its input to increase both the selectivity and the
invariance of the representation. With multiple non-linear layers, say
a depth of 5 to 20, a system can implement extremely intricate func-
tions of its inputs that are simultaneously sensitive to minute details
— distinguishing Samoyeds from white wolves — and insensitive to
large irrelevant variations such as the background, pose, lighting and
surrounding objects._

### Backpropagation to train multilayer architectures 
![Screenshot from 2019-10-14 23-10-11.png](https://i.loli.net/2019/10/16/pICHxlbNjvskyQg.png)[2]
![Screenshot from 2019-10-12 17-08-42.png](https://i.loli.net/2019/10/16/MfDBzXS6wiaEqnP.png)[1]


### Convolutional neural networks

_Many data modalities are in the form of multiple arrays: 1D for signals and
sequences, including language; 2D for images or audio spectrograms;
and 3D for video or volumetric images. There are four key ideas
behind ConvNets that take advantage of the properties of natural
signals: local connections, shared weights, pooling and the use of
many layers._

## 2. Related questions and discussion 

#### Q1. What is bias and variance in machine learning?  How to solve the problem of under-fitting and over-fitting?
![Screenshot from 2019-10-15 22-30-49.png](https://i.loli.net/2019/10/16/OPiz863yUqhIKxG.png) [3]

(1) In supervised learning, underfitting often take place when the model is too simple to capture the actual charateristic of the data. Normally it happens when there is not much data for training. It normally has a low variance but high bias.
On the other hand, overfitting happens when the model fit too well for a specific dataset (e.g. training data) but not generalize enough for other datasets. It usually has low bias but high variance. Overfitting often take place when the model
has been built too complicated. Therefore from this perspective, a good way to mitigate overfitting is to decrease the number of features (e.g. Drop-out); or use regularization to avoid certain weight parameters been too "powerful"*[]: 

(2) 
#### Q2. How to calculate the number of parameters in the CNN?
      (1) For convolution layers, suppose input feature map is a*b*l, output feature map is c*d*k, the kernel itself is m*n:
        num_of_params=(m*n*l+1)*k
        if no padding:
             a-n+1=c
             b-m+1=d
       (2) For fully connected layers, suppose input is n and output is m:
          num_of_params=(n+1)*m


#### Q3. Why use non-linear activation function? What will happen if use a linear activation function?

(1) Non-linear activation functions allow the model to create complex mappings between input and output data, which is suitable for modelling complex and high dimension data, such as image, video, audio etc. 

(2) _Two consequences:
-Not possible to use backpropagation  since the derivative is a constant and not related to the input. 
-All layers of the neural network collapse into one—with linear activation functions, no matter how many layers in the neural network, the last layer will be a linear function of the first layer (because a linear combination of linear functions is still a linear function).
  So a linear activation function turns the neural network into just one layer. A neural network with a linear activation function is simply a linear regression model. It has limited power and ability to handle complexity varying parameters of input data [5]._

#### Q4. What is gradient explosion and vanishing? How to deal with it?
(1) During the training process of a deep learning nn, the gradients back-propagated through the network from the output layer all the way to the initial layers, the gradients can therefore accumulate via matrix multiplications.
If they have small values (such as the derivative of an activation function<1), it will lead to an exponential shrink (vanish), resulting in weights unable to learn/update;
On the contrary, if they have big values (>1) it will lead to an explosive grow of the gradients on the way back and very large updates on the weights. The network will therefore become very unstable and unable to learn.  Typical signs:<br>
-Loss changes dramatically during updates;<br>
-Loss not decrease; <br>
-Weight parameters become big->NAN; <br>
-Loss become NAN. <br>

(2) For gradient explosion, fix methods:
-Regularization
-Gradient Clipping
-RELU saves the world lol
-Short cut in deep NN (ResNet)
-Using a shallower network

For gradient vanishing, fix methods:
-Chosing a more suitable activation function (e.g. using RELU to replace sigmoid)
-Modify weight initialization



## Reference
[1] LeCun, Yann & Bengio, Y. & Hinton, Geoffrey. (2015). Deep Learning. Nature. 521. 436-44. 10.1038/nature14539. <br>
[2] http://www.deepshare.net/ <br>
[3] https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229 <br>
[4] https://chemicalstatistician.wordpress.com/2014/03/19/machine-learning-lesson-of-the-day-overfitting-and-underfitting/ <br>
[5] https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/ <br>
