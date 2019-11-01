---
layout:     post
title:      Model Evaluation Metrics
subtitle:   Classification and Regression
date:       2019-10-22
author:     Mary Li
header-img: img/2019-10-12-bg.jpg
catalog: true
tags: 
    -  deep learning
---
_This post intends to summarise the evaluation metrics for both classification and regression models_

[![KT2soQ.md.png](https://s2.ax1x.com/2019/11/01/KT2soQ.md.png)](https://imgchr.com/i/KT2soQ)
<center>[1]</center>
## Classification 

![K854BV.png](https://s2.ax1x.com/2019/10/22/K854BV.png)

Recall=True Positive/(True Positive+False Negative)=TPR <br>
Precision=True Positive/(True Positive+False Positive) <br>
Accuracy=(True Positive +True Negative)/(True Positive+False Positive+True Negative+False Negative) <br>
Specify=True Negative/(False Positive+True Negative) <br>
FPR=1- specify=False Positive/(False Positive+True Negative) <br>

### 1. F1 Score, AP (average precision) and mAP
While ideally both recall and precision are high, in reality there is always a trade-off between the two.

![K8q2IP.png](https://s2.ax1x.com/2019/10/22/K8q2IP.png)
<center> [1] </center>
e.g. 
we predict 1 if h(x)>=threshold, and 0 otherwise.<br>
How to pick a good threshold for h(x) so that the classifier work best for us?<br><br>

-If recall is more important, lower the threshold;<br>
-If precision is more important, higher the threshold;<br>
-If we want a balanced measure, taking into account of these two factors equally: F1 score <br>

F1 score=2*(precision*recall)/(precision+recall) <br>
F1 ->1 ideally; F->0 worst case. <br>

The average precision (AP) is to find the area underneath the precision-recall curve (curve A and B shown 
above. Normally any zigzags in this curve are removed.  

A concept mAP (mean average precision) is often used as the average of AP. 

### 2. ROC
In reality, if you want more positives been predicted in your model, the rate of false positives will increase too.
ROC Curve is used to describe the trade-off between TPR and FPR:

![K87orR.png](https://s2.ax1x.com/2019/10/22/K87orR.png)
<center> [1] </center>
Like AP, we use the area underneath the curve as a measure. In this case the bigger the better.

## 2. Regression
[![KThklR.md.png](https://s2.ax1x.com/2019/11/01/KThklR.md.png)](https://imgchr.com/i/KThklR)

## Reference
[1] "Getting Started with Machine Learning" by Jim Liang <br>