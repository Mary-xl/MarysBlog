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
_From this week, I will begin a paper digest series in which classic papers are read, extracted and discussed with related questions and thought.
 This article is the digest from [1]_

## 1. Excerpt
To properly adjust the weight vector, the learning algorithm computes a gradient vector that, for each weight, indicates by what amount
the error would increase or decrease if the weight were increased by a tiny amount. The weight vector is then adjusted in the opposite direction to the gradient vector.
![Screenshot from 2019-10-09 21-58-07.png](https://i.loli.net/2019/10/16/LC3pmPNd6a95yB8.png)[2]

The objective function, averaged over all the training examples, can be seen as a kind of hilly landscape in the high-dimensional space of
weight values. The negative gradient vector indicates the direction
of steepest descent in this landscape, taking it closer to a minimum,
where the output error is low on average.

A deep-learning architecture is a multilayer stack of simple mod-
ules, all (or most) of which are subject to learning, and many of which
compute non-linear input–output mappings. Each module in the
stack transforms its input to increase both the selectivity and the
invariance of the representation. With multiple non-linear layers, say
a depth of 5 to 20, a system can implement extremely intricate func-
tions of its inputs that are simultaneously sensitive to minute details
— distinguishing Samoyeds from white wolves — and insensitive to
large irrelevant variations such as the background, pose, lighting and
surrounding objects.

### Backpropagation to train multilayer architectures 
![Screenshot from 2019-10-14 23-10-11.png](https://i.loli.net/2019/10/16/pICHxlbNjvskyQg.png)[2]
![Screenshot from 2019-10-12 17-08-42.png](https://i.loli.net/2019/10/16/MfDBzXS6wiaEqnP.png)[1]


### Convolutional neural networks

Many data modalities are in the form of multiple arrays: 1D for signals and
sequences, including language; 2D for images or audio spectrograms;
and 3D for video or volumetric images. There are four key ideas
behind ConvNets that take advantage of the properties of natural
signals: local connections, shared weights, pooling and the use of
many layers.

## 2. Related questions and discussion 

#### Q1. What is bias and variance in machine learning?  How to solve the problem of under-fitting and over-fitting?
![Screenshot from 2019-10-15 22-30-49.png](https://i.loli.net/2019/10/16/OPiz863yUqhIKxG.png) [3]
In supervised learning, underfitting happens when a model unable to capture the underlying pattern of the data. These models usually have high bias and low variance. It happens when we have very less amount of data to build an accurate model or when we try to build a linear model with a nonlinear data. Also, these kind of models are too simple to capture the complex patterns in data[3].
Intuitively, overfitting occurs when the model or the algorithm fits the data too well.  Specifically, overfitting occurs if the model or algorithm shows low bias but high variance.  Overfitting is often a result of an excessively complicated model, and it can be prevented by fitting multiple models and using validation or cross-validation to compare their predictive accuracies on test data [4].

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

(2) Two consequences:
-Not possible to use backpropagation  since the derivative is a constant and not related to the input. 
 -All layers of the neural network collapse into one—with linear activation functions, no matter how many layers in the neural network, the last layer will be a linear function of the first layer (because a linear combination of linear functions is still a linear function). So a linear activation function turns the neural network into just one layer. A neural network with a linear activation function is simply a linear regression model. It has limited power and ability to handle complexity varying parameters of input data [5].

## Reference
[1] LeCun, Yann & Bengio, Y. & Hinton, Geoffrey. (2015). Deep Learning. Nature. 521. 436-44. 10.1038/nature14539. <br>
[2] http://www.deepshare.net/ <br>
[3] https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229 <br>
[4] https://chemicalstatistician.wordpress.com/2014/03/19/machine-learning-lesson-of-the-day-overfitting-and-underfitting/ <br>
[5] https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/ <br>
