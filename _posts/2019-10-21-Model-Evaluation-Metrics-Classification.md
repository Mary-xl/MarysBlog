---
layout:     post
title:      Model Evaluation Metrics
subtitle:   Classification
date:       2019-10-22
author:     Mary Li
header-img: img/
catalog: true
tags: 
    -  deep learning
---
_This article intend to summarise the evaluation metrics for classification models_

![K854BV.png](https://s2.ax1x.com/2019/10/22/K854BV.png)

Recall=True Positive/(True Positive+False Negative)=TPR <br>
Precision=True Positive/(True Positive+False Positive) <br>
Accuracy=(True Positive +True Negative)/(True Positive+False Positive+True Negative+False Negative) <br>
Specify=True Negative/(False Positive+True Negative) <br>
FPR=1- specify=False Positive/(False Positive+True Negative) <br>

## Q1: AP and F1 Score
While ideally both recall and precision are high, in reality there is always a trade-off between the two.

![K8q2IP.png](https://s2.ax1x.com/2019/10/22/K8q2IP.png)

e.g. 
we predict 1 if h(x)>=threshold, and 0 otherwise.<br>
How to pick a good threshold for h(x) so that the classifier work best for us?<br><br>

-If recall is more important, lower threshold;<br>
-If precision is more important, higher threshold;<br>
-If we want a balanced measure, taking into account of these two factors equally: F1 score <br>

F1 score=2*(precision*recall)/(precision+recall) <br>
F1 ->1 ideally; F->0 worst case.

## Q2: ROC
In reality, if you want more positives been predicted in your model, the rate of false positives will increase too.
ROC Curve is used to describe the trade-off between TPR and FPR:

![K87orR.png](https://s2.ax1x.com/2019/10/22/K87orR.png)

We use the area underneath the curve as a measure. In this case the bigger the better.

## Reference
-"Getting Started with Machine Learning" by Jim Liang