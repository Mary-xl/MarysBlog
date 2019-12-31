[TOC]

# 一. 建议先阅读以下资料，然后再去看代码

代码GitHub地址：

https://github.com/chenyuntc/simple-faster-rcnn-pytorch (数据、权重下载链接和训练方法) 

 

Faster R-CNN博客：

https://zhuanlan.zhihu.com/p/32404424 (本代码是这个作者写的，讲解与代码相匹配)

 

Faster R-CNN论文中英翻译 ：

https://blog.csdn.net/quincuntial/article/details/79132243 

 

Faster R-CNN知识讲解.md  (按照代码顺序 结合了论文、代码、博客)



# 二.如何阅读代码注释

[]括号代表注释源自于哪里。 

[fan]：代表助教注释 

无注释：代表代码中原本就有的 

[0]https://zhuanlan.zhihu.com/p/32404424

[1]https://blog.csdn.net/qq_32678471/article/details/84776144

[2]https://www.cnblogs.com/kerwins-AC/p/9734381.html 

[3]https://blog.csdn.net/xiewenbo/article/details/78320796 

[4]https://www.cnblogs.com/king-lps/p/8981222.html 

[5]https://zhuanlan.zhihu.com/p/52625664 



# 三.数据集介绍

Pascal VOC 2007  
The goal of this challenge is to recognize objects from a number of visual object classes in realistic scenes (i.e. not pre-segmented objects). It is fundamentally a supervised learning learning problem in that a training set of labelled images is provided. The twenty object classes that have been selected are:  

Person: person  
Animal: bird, cat, cow, dog, horse, sheep  
Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train  
Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor  

Main Competitions   (我们这里是目标检测)
1.Classification: For each of the twenty classes, predicting presence/absence of an example of that class in the test image.  
2.Detection: Predicting the bounding box and label of each object from the twenty target classes in the test image. 

The training data provided consists of a set of images; each image has an annotation file giving a bounding box and object class label for each object in one of the twenty classes present in the image. Note that multiple objects from multiple classes may be present in the same image.   

The data has been split into 50% for training/validation and 50% for testing. The distributions of images and objects by class are approximately equal across the training/validation and test sets. In total there are 9,963 images, containing 24,640 annotated objects.   

http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/index.html  

这个数据集是Faster R-CNN论文中主要评估的数据集，也是这份代码所使用的数据集，总共大小约1G。  



# 四.代码流程

训练入口函数：train.py  def train(**kwargs)

设置超参： utils/config.py  class Config:

训练数据集设置： data/dataset.py   class Dataset

测试数据集设置： data/dataset.py   class TestDataset

构造模型:  model/faster_rcnn_vgg16.py  model/faster_rcnn_vgg16.py  

设置前向传播/数据流：trainer.py  class FasterRCNNTrainer(nn.Module):

​		特征提取：model/faster_rcnn_vgg16.py  def decom_vgg16():  

​		RPN: model/region_proposal_network.py  class RegionProposalNetwork(nn.Module): def forward

​		NMS:model/utils/nms/non_maximum_suppression.py    def _non_maximum_suppression_gpu

​		2000个候选框筛选128个： model/utils/creator_tool.py  class ProposalTargetCreator:   def __call__

​		RoI poolling:  model/roi_module.py    class RoI(Function):     def forward(self, x, rois):

​		RoI Head/Fast R-CNN: model/faster_rcnn_vgg16.py model/faster_rcnn_vgg16.py def forward

​		RPN losses: trainer.py  class FasterRCNNTrainer(nn.Module):  def forward

​		ROI losses: trainer.py  class FasterRCNNTrainer(nn.Module):  def forward

训练(执行 前向传播、反向传播、优化):  trainer.py   def train_step

验证评估 :  train.py    def eval



# 五.Faster R-CNN知识讲解(按照代码前向执行顺序)

## 1.Faster R-CNN描述
在目标检测领域, Faster R-CNN表现出了极强的生命力, 虽然是2015年的论文, 但它至今仍是许多目标检测算法的基础，这在日新月异的深度学习领域十分难得。Faster R-CNN还被应用到更多的领域中, 比如人体关键点检测、目标追踪、 实例分割还有图像描述等。  



论文中描述  ：

在ILSVRC和COCO 2015竞赛中，Faster R-CNN和RPN是ImageNet检测，ImageNet定位，COCO检测和COCO分割中几个第一名参赛者[18]的基础。  
In ILSVRC and COCO 2015 competitions, Faster R-CNN and RPN are the basis of several 1st-place entries [18] in the tracks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation. RPNs completely learn to propose regions from data, and thus can easily benefit from deeper and more expressive features (such as the 101-layer residual nets adopted in [18]).



## 2.Faster R-CNN整体架构  
### 2.1.从代码角度进行描述
![](./image/0.png)  从编程角度来说， Faster R-CNN主要分为四部分（图中四个绿色框）：

- Dataset：数据，提供符合要求的数据格式（目前常用数据集是VOC和COCO）
- Extractor： 利用CNN提取图片特征`features`（原始论文用的是ZF和VGG16，后来人们又用ResNet101）
- RPN(*Region Proposal Network):* 负责提供候选区域`RoIs`（每张图给出大概2000个候选框）
- RoIHead： 负责对`rois`分类和微调。对RPN找出的`RoIs`，判断它是否包含目标，并修正框的位置和座标

Faster R-CNN整体的流程可以分为三步：

- 提特征： 图片（`img`）经过预训练的网络（`Extractor`），提取到了图片的特征（`feature`）
- Region Proposal： 利用提取的特征（`feature`），经过RPN网络，找出一定数量的`RoIs`（region of interests）。
- 分类与回归：将`RoIs`和图像特征`features`，输入到`RoIHead`，对这些`RoIs`进行分类，判断都属于什么类别，同时对这些`RoIs`的位置进行微调。

![](image/28.jpg)



### 2.2.论文中的描述：  
我们的目标检测系统，称为Faster R-CNN，由两个模块组成。第一个模块是提议区域的深度全卷积网络，第二个模块是使用提议区域的Fast R-CNN检测器。整个系统是一个单个的，统一的目标检测网络（下图）。使用最近流行的“注意力”机制的神经网络术语，RPN模块告诉Fast R-CNN模块在哪里寻找。  

Our object detection system, called Faster R-CNN, is composed of two modules. The first module is a deep fully convolutional network that proposes regions, and the second module is the Fast R-CNN detector [2] that uses the proposed regions. The entire system is a single, unified network for object detection (Figure 2). Using the recently popular terminology of neural networks with attention [31] mechanisms, the RPN module tells the Fast R-CNN module where to look.   

![671D8F15-0C3B-4C06-A375-9058345B9652.png](./image/2.png)

补充说明 上面介绍的 Fast R-CNN的架构图  (来自七月课件)，忽略图中的SS(region proposal)部分，即为上面描述的Fast R-CNN模块:

![F6B99A2B-1D9E-4934-A521-7E50C971182D.png](./image/3.png)

补充：

两句话介绍Faster R-CNN的两阶段内容(来自Mask R-CNN论文)：

Faster R-CNN由两个阶段组成。称为区域提议网络（RPN）的第一阶段提出候选目标边界框。第二阶段，本质上是Fast R-CNN ，使用RoIPool从每个候选框中提取特征，并进行分类和边界回归。
Faster R-CNN consists of two stages.The first stage, called a Region Proposal Network (RPN),proposes candidate object bounding boxes. The second stage, which is in essence Fast R-CNN [12], extracts features using RoIPool from each candidate box and performs classification and bounding-box regression.





## 3.超参设置
代码： utils/config.py  



## 4.数据预处理  
代码： data/dataset.py  class Transform(object):  

图片进行缩放，使得长边小于等于1000，短边小于等于600（至少有一个等于）。  
代码： data/dataset.py  def(preprocess)  

对相应的bounding boxes 也也进行同等尺度的缩放  
代码： data/util.py  def resize_bbox(bbox, in_size, out_size):  



## 5.Extractor  

取VGG-16 前13层用于特征提取，为了节省显存，前四层卷积层的学习率设为0  
代码： 
model/faster_rcnn_vgg16.py  def decom_vgg16():  



在我们的实验中，我们研究了具有5个共享卷积层的Zeiler和Fergus模型（ZF）和具有13个共享卷积层的Simonyan和Zisserman模型（VGG-16）。

In our experiments, we investigate the Zeiler and Fergus model 32, which has 5 shareable convolutional layers and the Simonyan and Zisserman model 3, which has 13 shareable convolutional layers.  


Conv5_3的输出作为图片特征（feature）。conv5_3相比于输入，下采样了16倍，也就是说输入的图片尺寸为3×H×W，那么feature的尺寸就是C×(H/16)×(W/16)。总之，一张图片，经过extractor之后，会得到一个C×(H/16)×(W/16)的feature map。
 ![5F4B392A-EC20-4929-A0C8-F79AC1A5A385.png](./image/4.png)

## 6.Region Proposal Network
### 6.1.RPN是什么？
Region Proposal Network(RPN)是一个全卷积网络，可以同时在每个位置预测 目标边界和目标分数(二分类)。 
RPN全称是Region Proposal Network区域提议网络， 它是区域提议Region Proposal的一种方法。

An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position.  

区域提议的方法还有SS,EB：  
![11C74E65-4B3A-4C98-9C54-4F89C01FD96A.png](image/5.png)  



### 6.2.RPN的作用？

RPN经过端到端的训练，可以生成高质量的区域提议。RPN组件告诉统一网络在哪里寻找。对于非常深的VGG-16模型，我们的检测系统在GPU上的帧率为5fps（包括所有步骤），同时在PASCAL VOC 2007，2012和MS COCO数据集上实现了最新的目标检测精度，每个图像只有300个提议。  
The RPN is trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. We further merge RPN and Fast R-CNN into a single network by sharing their convolutional features——using the recently popular terminology of neural networks with “attention” mechanisms, the RPN component tells the unified network where to look. For the very deep VGG-16 model [3], our detection system has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection accuracy on PASCAL VOC 2007, 2012, and MS COCO datasets with only 300 proposals per image.   



Faster R-CNN最突出的贡献就在于提出了Region Proposal Network（RPN）代替了Selective Search，从而将候选区域提取的时间开销几乎降为0（Fast R-CNN 2s   ->   Faster R-CNN 0.01s）, 见下面两幅图：

![A5235767-A6FB-4E8B-8812-21CB4134AF40.png](image/6.png)  
![6C5E7C17-B806-4563-AB4C-D8F89B1A5D7A.png](image/7.png)  



### 6.3.Anchor

#### 6.3.1.Anchor是什么?

anchor是提议框的初始化。RPN会在anchor的基础上回归微调。[fan]



在每个滑动窗口位置，我们同时预测多个区域提议，其中每个位置可能提议的最大数目表示为k。因此，reg层具有4k个输出，编码k个边界框的坐标，cls层输出2k个分数，估计每个提议是目标或不是目标的概率。相对于我们称之为锚点的k个参考边界框，k个提议是参数化的。锚点位于所讨论的滑动窗口的中心，并与一个尺度和长宽比相关。默认情况下，我们使用3个尺度和3个长宽比，在每个滑动位置产生k=9个锚点。对于大小为W\*H（通常约为2400）的卷积特征映射，总共有W\*H\*k个锚点。  
At each sliding-window location, we simultaneously predict multiple region proposals, where the number of maximum possible proposals for each location is denoted as k. So the reg layer has 4k outputs encoding the coordinates of k boxes, and the cls layer outputs 2k scores that estimate probability of object or not object for each proposal. The k proposals are parameterized relative to k reference boxes, which we call anchors. An anchor is centered at the sliding window in question, and is associated with a scale and aspect ratio (Figure 3, left). By default we use 3 scales and 3 aspect ratios, yielding k=9 anchors at each sliding position. For a convolutional feature map of a size W × H (typically ∼2,400), there are WHk anchors in total.  

在RPN中，作者提出了`anchor`。Anchor是大小和尺寸固定的候选框。论文中用到的anchor有三种尺寸(边长)和三种比例，如下图所示，三种尺寸分别是小（蓝128）中（红256）大（绿512），三个比例分别是1:1，1:2，2:1。3×3的组合总共有9种anchor。

![](image/32.jpg)

然后用这9种anchor在特征图（`feature`）左右上下移动，每一个特征图上的点都有9个anchor，最终生成了 (H/16)× (W/16)×9个`anchor`. 对于一个512×62×37的feature map，有 62×37×9~ 20000个anchor。 也就是对一张图片，有20000个左右的anchor。这种做法很像是暴力穷举，20000多个anchor，哪怕是蒙也能够把绝大多数的ground truth bounding boxes蒙中。

  

#### 6.3.2.为什么要引入anchor  ?

我们的锚点设计提出了一个新的方案来解决多尺度。  
如下图所示，多尺度预测有两种流行的方法。第一种方法是基于图像/特征金字塔，例如DPM和基于CNN的方法中。图像在多个尺度上进行缩放，并且针对每个尺度计算特征映射（HOG或深卷积特征 下图(a)）。这种方法通常是有用的，但是非常耗时。第二种方法是在特征映射上使用多尺度的滑动窗口。例如，在DPM中，使用不同的滤波器大小（例如5×7和7×5）分别对不同长宽比的模型进行训练。如果用这种方法来解决多尺度问题，可以把它看作是一个“滤波器金字塔”（下图（b））。第二种方法通常与第一种方法联合采用。  
作为比较，我们的基于锚点方法建立在锚点金字塔上，这是更具成本效益的。我们的方法参照多尺度和长宽比的锚盒来分类和回归边界框。它只依赖单一尺度的图像和特征映射，并使用单一尺寸的滤波器（特征映射上的滑动窗口）。  
Our design of anchors presents a novel scheme for addressing multiple scales (and aspect ratios). As shown in Figure 1, there have been two popular ways for multi-scale predictions. The first way is based on image/feature pyramids, e.g., in DPM [8] and CNN-based methods [9], [1], [2]. The images are resized at multiple scales, and feature maps (HOG [8] or deep convolutional features [9], [1], [2]) are computed for each scale (Figure 1(a)). This way is often useful but is time-consuming. The second way is to use sliding windows of multiple scales (and/or aspect ratios) on the feature maps. For example, in DPM [8], models of different aspect ratios are trained separately using different filter sizes (such as 5×7 and 7×5). If this way is used to address multiple scales, it can be thought of as a “pyramid of filters” (Figure 1(b)). The second way is usually adopted jointly with the first way [8].  
![A360F4B6-5421-4DF0-BC47-E12B58360DAE.png](/Users/fan/Downloads/fan/simple-faster-rcnn/image/11.png)

anchor实现多尺度只是取值，而图片和滤波器多尺度需要多次运算[fan]。  



生成9个anchor模板的代码：  
model/utils/bbox_tools.py  
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2]   

生成每个位置的9个anchor的代码：  
model/region_proposal_network.py  
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):  



### 6.4.如何实现RPN  ?

#### 6.4.1.RPN的模型设计
区域提议网络（RPN）以任意大小的图像作为输入，输出一组矩形的目标提议，每个提议都有一个目标得分(有无物体)。我们用全卷积网络对这个过程进行建模。因为我们的最终目标是与Fast R-CNN目标检测网络共享计算，所以我们假设两个网络共享一组共同的卷积层。  
A Region Proposal Network (RPN) takes an image (of any size) as input and outputs a set of rectangular object proposals, each with an objectness score.3 We model this process with a fully convolutional network [7], which we describe in this section. Because our ultimate goal is to share computation with a Fast R-CNN object detection network [2], we assume that both nets share a common set of convolutional layers.  

为了生成区域提议，我们在最后的共享卷积层输出的卷积特征映射上滑动一个小网络。这个小网络将输入卷积特征映射的n×n空间窗口作为输入。每个滑动窗口映射到一个低维特征（ZF为256维，VGG为512维，后面是ReLU）。这个特征被输入到两个子全连接层——一个边界框回归层（reg）和一个边界框分类层（cls）。在本文中，我们使用n=3。下图(左）显示了这个小型网络的一个位置。请注意，因为小网络以滑动窗口方式运行，所有空间位置共享全连接层。这种架构通过一个n×n卷积层，后面是两个子1×1卷积层（分别用于reg和cls）自然地实现。  
To generate region proposals, we slide a small network over the convolutional feature map output by the last shared convolutional layer. This small network takes as input an n×n spatial window of the input convolutional feature map. Each sliding window is mapped to a lower-dimensional feature (256-d for ZF and 512-d for VGG, with ReLU [33] following). This feature is fed into two sibling fully-connected layers——a box-regression layer (reg) and a box-classification layer (cls). We use n=3 in this paper, noting that the effective receptive field on the input image is large (171 and 228 pixels for ZF and VGG, respectively). This mini-network is illustrated at a single position in Figure 3 (left). Note that because the mini-network operates in a sliding-window fashion, the fully-connected layers are shared across all spatial locations. This architecture is naturally implemented with an n×n convolutional layer followed by two sibling 1 × 1 convolutional layers (for reg and cls, respectively).  
![E8E4AB36-CEDC-40A3-BD2C-BAA7737408B8.png](image/8.png)



RPN在`Extractor`输出的feature maps的基础之上，先增加了一个卷积，然后利用两个1x1的卷积分别进行二分类（是否为正样本）和位置回归。进行分类的卷积核通道数为9×2（9个anchor，每个anchor二分类，使用交叉熵损失），进行回归的卷积核通道数为9×4（9个anchor，每个anchor有4个位置参数）。RPN是一个全卷积网络（fully convolutional network），这样对输入图片的尺寸就没有要求了。

![645E662A-FC1C-4188-B813-E54308BE694D.png](image/9.png)

代码：  
model/region_proposal_network.py  class RegionProposalNetwork(nn.Module):  def __init__



#### 6.4.2.如何训练RPN  ?

因为数据集只提供物体的边界框和类别标签，没有提供anchor的类别和回归偏移量，所以为了训练RPN，需要我们自己定义正负样本和回归真值[fan]。



对所有锚点的损失函数进行优化是可能的，但是这样会偏向于负样本，因为它们是占主导地位的。取而代之的是，我们在图像中随机采样256个锚点，计算一个小批量数据的损失函数，其中采样的正锚点和负锚点的比率可达1:1。如果图像中的正样本少于128个，我们使用负样本填充小批量数据

The RPN can be trained end-to-end by back-propagation and stochastic gradient descent (SGD) [35]. We follow the “image-centric” sampling strategy from [2] to train this network. Each mini-batch arises from a single image that contains many positive and negative example anchors. It is possible to optimize for the loss functions of all anchors, but this will bias towards negative samples as they are dominate. Instead, we randomly sample 256 anchors in an image to compute the loss function of a mini-batch, where the sampled positive and negative anchors have a ratio of up to 1:1. If there are fewer than 128 positive samples in an image, we pad the mini-batch with negative ones.



##### (a).划分正负样本

论文中描述：

为了训练RPN，我们为每个锚点分配一个二值类别标签（是目标或不是目标）。

我们给两种锚点分配一个正标签(代码中的label值为1)：（i）具有与实际边界框的重叠最高交并比（IoU）的锚点，或者（ii）具有与实际边界框的重叠超过0.7 IoU的锚点。注意，单个真实边界框可以为多个锚点分配正标签。通常第二个条件足以确定正样本；但我们仍然采用第一个条件，因为在一些极少数情况下，第二个条件可能找不到正样本。

对于所有的真实边界框，如果一个锚点的IoU比率低于0.3，我们给此锚点分配一个负标签(代码中的label值为0)。既不正也不负的锚点(代码中的label值为-1)不会有助于训练目标函数。  
For training RPNs, we assign a binary class label (of being an object or not) to each anchor. We assign a positive label to two kinds of anchors: (i) the anchor/anchors with the highest Intersection-over-Union (IoU) overlap with a ground-truth box, or (ii) an anchor that has an IoU overlap higher than 0.7 with any ground-truth box. Note that a single ground-truth box may assign positive labels to multiple anchors. Usually the second condition is sufficient to determine the positive samples; but we still adopt the first condition for the reason that in some rare cases the second condition may find no positive sample. We assign a negative label to a non-positive anchor if its IoU ratio is lower than 0.3 for all ground-truth boxes. Anchors that are neither positive nor negative do not contribute to the training objective.  



博客中描述：

接下来RPN做的事情就是利用（`AnchorTargetCreator`）将20000多个候选的anchor选出256个anchor进行分类和回归位置。选择过程如下：

- 对于每一个ground truth bounding box (`gt_bbox`)，选择和它重叠度（IoU）最高的一个anchor作为正样本
- 对于剩下的anchor，从中选择和任意一个`gt_bbox`重叠度超过0.7的anchor，作为正样本，正样本的数目不超过128个。
- 随机选择和`gt_bbox`重叠度小于0.3的anchor作为负样本。负样本和正样本的总数为256。

对于每个anchor, gt_label 要么为1（前景），要么为0（背景），而gt_loc则是由4个位置参数(tx,ty,tw,th)组成，这样比直接回归座标更好。



代码:设置RPN的正负样本和计算对应分类真值

model/utils/creator_tool.py

class AnchorTargetCreator(object):

def __call__(self, bbox, anchor, img_size):



##### (b).RPN多任务中的分类loss

$$
\frac{1}{N_{c l s}} \sum_{i} L_{c l s}\left(p_{i}, p_{i}^{*}\right)
$$

其中：

​	$$L_{c l s}\left(p_{i}, p_{i}^{*}\right)$$是两个类别(目标 vs非目标)的对数损失：
$$
L_{c l s}\left(p_{i}, p_{i}^{*}\right)=-\log \left[p_{i}^{*} p_{i}+\left(1-p_{i}^{*}\right)\left(1-p_{i}\right)\right]
$$
可以看到这是一个经典的二分类**交叉熵损失**，对于每一个anchor计算对数损失，然后求和除以总的anchor数量Ncls。在训练RPN的阶段，Ncls = 256，在训练fast rcnn的阶段，Ncls = 128。

$$p_{i}$$为$$anchor$$预测为目标的概率；

$$GT$$标签:
$$
p_{i}^{*}=\left\{\begin{array}{ll}{0} & {\text { negative label }} \\ {1} & {\text { positive label }}\end{array}\right.
$$




实际代码中直接调用了PyTorch库

trainer.py

class FasterRCNNTrainer(nn.Module):

def forward(self, imgs, bboxes, labels, scale):

```python
from torch.nn import functional as F
RPN_cls_loss = F.cross_entropy(RPN_score, gt_RPN_label.cuda(), ignore_index=-1) #RPN_score为RPN网络得到的（大约20000个）与anchor_target_creator得到的2000个label求交叉熵损失 [1] RPN分类损失 公式：  https://www.cnblogs.com/marsggbo/p/10401215.html   第一个参数是每个anchor前景和背景的概率，第二个参数是这个anchor的类别(-1,0,1)，第三个参数代表忽略target是-1的anchor，最后默认会求平均(对target为1或0的anchor求平均) [fan]
```



##### (c).RPN多任务中的回归loss

Ground truth位置(G_hat)与anchor位置(P)的变化关系(公式源自于RCNN，所以先用RCNN版本来解释)：
$$
\begin{array}{l}{\hat{G}_{x}=P_{w} d_{x}(P)+P_{x}} \\ {\hat{G}_{y}=P_{h} d_{y}(P)+P_{y}} \\ {\hat{G}_{w}=P_{w} \exp \left(d_{w}(P)\right)} \\ {\hat{G}_{h}=P_{h} \exp \left(d_{h}(P)\right)}\end{array}
$$
在RCNN中，利用**class-specific**（特定类别）的bounding box regressor。也即每一个类别学一个回归器，然后对该类的bbox预测结果进行进一步微调。注意在回归的时候要将bbox坐标(左上右下)转为中心点(x,y)与宽高(w,h)。对于bbox的预测结果P和gt_bbox 来说我们学要学一个变换，使得这个变换可以将P映射到一个新的位置，使其尽可能逼近gt_bbox。

这四个参数(dx,dy,dw,dh)都是特征的函数，前两个体现为bbox的中心尺度不变性，后两个体现为体现为bbox宽高的对数空间转换。学到这四个参数（函数）后，就可以将P映射到G', 使得G'尽量逼近ground truth G



回归预测值：(源自于RCNN，所以先用RCNN版本来解释)：

预测的是从anchor到ground truth的变化函数



那么这个参数组 $$d_{*}(P)$$是怎么得到的呢？它是关于候选框P的pool5 特征的函数。由pool5出来的候选框P的特征我们定义为$$\phi_{5}(P)$$，那么我们有：$$d_{\star}(P)=\mathbf{w}_{\star}^{\mathrm{T}} \phi_{5}(P)$$

其中W就是可学习的参数。也即我们要学习的参数组 $$d_{*}(P)$$等价于W与特征的乘积。那么回归的目标参数组是什么呢？就是上面四个式子中的逆过程：
$$
\begin{array}{l}{t_{x}=\left(G_{x}-P_{x}\right) / P_{w}} \\ {t_{y}=\left(G_{y}-P_{y}\right) / P_{h}} \\ {t_{w}=\log \left(G_{w} / P_{w}\right)} \\ {t_{h}=\log \left(G_{h} / P_{h}\right)}\end{array}
$$


回归预测值 和 回归真值(从RCNN 对应到 Faster R-CNN)：
$$
\begin{aligned} t_{\mathrm{x}} &=\left(x-x_{\mathrm{a}}\right) / w_{\mathrm{a}}, \quad t_{\mathrm{y}}=\left(y-y_{\mathrm{a}}\right) / h_{\mathrm{a}} \\ t_{\mathrm{w}} &=\log \left(w / w_{\mathrm{a}}\right), \quad t_{\mathrm{h}}=\log \left(h / h_{\mathrm{a}}\right) \\ t_{\mathrm{x}}^{*} &=\left(x^{*}-x_{\mathrm{a}}\right) / w_{\mathrm{a}}, \quad t_{\mathrm{y}}^{*}=\left(y^{*}-y_{\mathrm{a}}\right) / h_{\mathrm{a}} \\ t_{\mathrm{w}}^{*} &=\log \left(w^{*} / w_{\mathrm{a}}\right), \quad t_{\mathrm{h}}^{*}=\log \left(h^{*} / h_{\mathrm{a}}\right) \end{aligned}
$$
其中x,y,w,h分别为bbox的中心点坐标，宽与高。$$
x, x_{a}, x^{*}$$分别是预测box、anchor box、真实box。计算类似于RCNN，前两行是预测的box关于anchor的offset与scales，后两行是真实box与anchor的offset与scales。那回归的目的很明显，即使得$$t_{i}、t_{i}^*$$尽可能相近。回归损失函数利用的是Fast-RCNN中定义的Smooth L1函数，对外点更不敏感：

(以上参考链接：https://www.cnblogs.com/king-lps/p/8981222.html)

以上回归公式可参考论文：

Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation     的3.5小节 Bounding-box regression




$$
\lambda \frac{1}{N_{r e g}} \sum_{i} p_{i}^{*} L_{r e g}\left(t_{i}, t_{i}^{*}\right)
$$
其中：

$$p_{i}^{*}=\left\{\begin{array}{ll}{0} & {\text { negative label }} \\ {1} & {\text { positive label }}\end{array}\right. $$

$$t_{i}=\left\{t_{x}, t_{y}, t_{w}, t_{h}\right\}$$是一个向量，表示该anchor**预测的偏移量**。

$$t_{i}^{*}$$是与$$t_{i}$$维度相同的向量，表示anchor相对于gt**实际的偏移量**
$$
L_{r e g}\left(t, t_{i}^{*}\right)=R\left(t_{i}-t_{i}^{*}\right)
$$
R是smoothL1 函数，就是我们上面说的，不同之处是这里σ = 3， $$x = t_{i} - t_{i}^{*}$$
$$
\operatorname{smooth}_{L_{1}}(x)=\left\{\begin{array}{l}{0.5 x^{2} \times 1 / \sigma^{2} \text { if }|x|<1 / \sigma^{2}} \\ {|x|-0.5 \text { otherwise }}\end{array}\right.
$$
对于每一个anchor 计算完$$L_{r e g}\left(t_{i}, t_{i}^{*}\right)$$部分后还要乘以$$P^{*}$$，如前所述，$$P^{*}$$有物体时（positive）为1，没有物体（negative）时为0，意味着只有前景才计算损失，背景不计算损失。inside_weights就是这个作用。

对于$$\lambda$$和Nreg的解释在RPN训练过程中如下（之所以以RPN训练为前提因为此时batch size = 256，如果是fast rcnn，batchsize = 128）：

论文中，$$N_{reg}$$是feature map的size (约2400，600*1000的图片),  然后将$$\lambda$$取为10，这样分类和回归两个loss的权重基本相同( $$N_{cls}$$为256， 即batch size, 后面的 $$N_{reg} / \lambda$$约为240，二者接近)。

而代码中，直接将$$N_{reg}$$取为正负样本的总数(256, batch size) ，然后$$\lambda$$取为1， 这样相当于直接使分类和回归的loss权重相同，都为1/(batch size)，不需要额外设置$$\lambda$$了。

代码中，RPN的回归损失公式中 $$\sigma$$参数为3，Faster R-CNN的回归损失公式中 $$\sigma$$参数为1



补充：为什么要使用Smoooh L1 Loss 而不是 L2损失 ？

对于边框的预测是一个回归问题。通常可以选择平方损失函数（L2损失）f(x)=x^2。但这个损失对于比较大的误差的惩罚很高。

我们可以采用稍微缓和一点绝对损失函数（L1损失）f(x)=|x|，它是随着误差线性增长，而不是平方增长。但这个函数在0点处导数不存在，因此可能会影响收敛。

一个通常的解决办法是，分段函数，在0点附近使用平方函数使得它更加平滑。它被称之为平滑L1损失函数。它通过一个参数σ 来控制平滑的区域。
$$
\operatorname{smooth}_{L_{1}}(x)=\left\{\begin{array}{l}{0.5 x^{2} \times 1 / \sigma^{2} \text { if }|x|<1 / \sigma^{2}} \\ {|x|-0.5 \text { otherwise }}\end{array}\right.
$$
![](image/31.png)

(以上资料参考:https://blog.csdn.net/Mr_health/article/details/84970776)



生成每个anchor 回归真值的代码：

model/utils/bbox_tools.py

def bbox2loc(src_bbox, dst_bbox):



Smoooh L1 Loss代码：

trainer.py

def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):

def _smooth_l1_loss(x, t, in_weight, sigma):





##### (d).RPN多任务loss

根据这些定义，我们对目标函数Fast R-CNN[2]中的多任务损失进行最小化。我们对图像的损失函数定义为：  
$$
\begin{array}{c}{L\left(\left\{p_{i}\right\},\left\{t_{i}\right\}\right)=\frac{1}{N_{c l s}} \sum_{i} L_{c l s}\left(p_{i}, p_{i}^{*}\right)} \\ {+\lambda \frac{1}{N_{r e g}} \sum_{i} p_{i}^{*} L_{r e g}\left(t_{i}, t_{i}^{*}\right)}\end{array}
$$
其中，$$i$$ 是一个小批量数据中锚点的索引，$$p_{i}$$ 是锚点$$i$$ 作为目标的预测概率。如果锚点为正，真实标签$$p_{i}^{*}$$ 为1，如果锚点为负，则为0。$$t_{i}$$是表示预测边界框4个参数化坐标的向量，而 $$ t_{i}^{*} $$ 是与正锚点相关的真实边界框的向量。分类损失$$L_{cls}$$是两个类别上（目标或不是目标）的对数损失。对于回归损失，我们使用$$
L_{r e g}\left(t_{i}, t_{i}^{*}\right)=R\left(t_{i}-t_{i}^{*}\right)
$$，其中$$R$$是鲁棒损失函数（平滑$$L1$$）。项$$p_{i}^{*} L_{reg}$$表示回归损失仅对于正锚点激活，否则被禁用（$$p_{i}^{*}$$=0）。*cls*和*reg*层的输出分别由{$$ p_{i}$$}和{$$t_{i}$$}组成。

Here, $$i$$  is the index of an anchor in a mini-batch and  $$p_{i}$$  is the predicted probability of anchor $$i$$  being an object. The ground-truth label $$p_{i}^{*}$$ is 1 if the anchor is positive, and is 0 if the anchor is negative. $$t_{i}$$ is a vector representing the 4 parameterized coordinates of the predicted bounding box, and $$ t_{i}^{*} $$ is that of the ground-truth box associated with a positive anchor. The classification loss $$L_{cls}$$ is log loss over two classes (object vs not object). For the regression loss, we use$$
L_{r e g}\left(t_{i}, t_{i}^{*}\right)=R\left(t_{i}-t_{i}^{*}\right)
$$ where $$R$$ is the robust loss function (smooth L1L1) defined in [2]. The term $$p_{i}^{*} L_{reg}$$ means the regression loss is activated only for positive anchors ($$p_{i}^{*}$$=1) and is disabled otherwise ($$p_{i}^{*}$$=0). The outputs of the *cls* and *reg* layers consist of {$$ p_{i}$$} and {$$t_{i}$$} respectively.

这两个项用Ncls和Nreg进行标准化，并由一个平衡参数λ加权。在我们目前的实现中（如在发布的代码中），方程中的cls项通过小批量数据的大小（即Ncls=256）进行归一化，reg项根据锚点位置的数量（即，Nreg∼24000）进行归一化。默认情况下，我们设置λ=10，因此*cls*和*reg*项的权重大致相等。我们通过实验显示，结果对宽范围的λ值不敏感。我们还注意到，上面的归一化不是必需的，可以简化。

The two terms are normalized by Ncls and Nreg and weighted by a balancing parameter λ. In our current implementation (as in the released code), the cls term is normalized by the mini-batch size (ie, Ncls=256) and the regreg term is normalized by the number of anchor locations (ie, Nreg∼2,400). By default we set λ=10, and thus both *cls* and *reg* terms are roughly equally weighted. We show by experiments that the results are insensitive to the values of λ in a wide range(Table 9). We also note that the normalization as above is not required and could be simplified.



在实际的RPN训练代码中，直接令λ/Nreg=1/Ncls=256


$$
\frac{\lambda}{N_{r e g}}=\frac{1}{b a t c h s i z e}=\left\{\begin{array}{ll}{\frac{1}{256},} & {RPN训练阶段} \\ {\frac{1}{128},} & { \text {Fast } {R-CNN阶段}}\end{array}\right.
$$



### 6.5.RPN生成RoIs

RPN在自身训练的同时，还会提供RoIs（region of interests）给Fast RCNN（RoIHead）作为训练样本。RPN生成RoIs的过程(`ProposalCreator`)如下：

- 通过得到的回归变换函数，修正anchor的位置，得到anchor 对应的 RoIs
- 根据得到的分类概率，选取是物体的概率较大的12000个anchor对应的RoIs
- 利用非极大值（(Non-maximum suppression, NMS）抑制，选出概率最大的2000个RoIs(后面一节分析)

注意：在inference的时候，为了提高处理速度，12000和2000分别变为6000和300.

注意：这部分的操作不需要进行反向传播，因此可以利用NumPy/Tensor实现。

RPN的输出：RoIs（形如2000×4或者300×4的tensor）



生成RoIs的代码：

model/utils/creator_tool.py

class ProposalCreator: 

def __call__

(non_maximum_suppression之前的代码部分)



### 6.6.NMS

#### 6.6.1.什么是NMS?

七月李博士的图：

![](/Users/fan/Downloads/fan/simple-faster-rcnn/image/25.png)

![](/Users/fan/Downloads/fan/simple-faster-rcnn/image/26.png)

![](/Users/fan/Downloads/fan/simple-faster-rcnn/image/27.png)



#### 6.6.2.为什么需要NMS?

一些RPN提议框互相之间高度重叠。为了减少冗余，我们在提议区域根据他们的*cls*分数采取非极大值抑制（NMS）。我们将NMS的IoU阈值固定为0.7，这就给每张图像留下了大约2000个提议区域。正如我们将要展示的那样(实验证明)，NMS不会损害最终的检测准确性，但会大大减少提议的数量。在NMS之后，我们使用top-N个排序了的提议区域来进行检测。

Some RPN proposals highly overlap with each other. To reduce redundancy, we adopt non-maximum suppression (NMS) on the proposal regions based on their *cls* scores. We fix the IoU threshold for NMS at 0.7, which leaves us about 2000 proposal regions per image. As we will show, NMS does not harm the ultimate detection accuracy, but substantially reduces the number of proposals. After NMS, we use the top-N ranked proposal regions for detection.



#### 6.6.3代码实现

##### (a). NMS的流程

假设所有提议框(RoIs)的集合为S，算法返回结果集合S’初始化为空集，具体算法如下：

1. 将S中所有框的置信度按照降序排列，选中最高分框B，并把该框从S中删除
2. 遍历S中剩余的框，如果和B的IoU大于一定阈值t，将其从S中删除
3. 从S未处理的框中继续选一个得分最高的加入到结果S‘中，重复上述过程1, 2。直至S集合空，此时S’即为所求结果。

算法整体看是比较简单而直接的，然而在实际执行过程中却不是这样，由于检测模型预测出来的框数量非常多，那么这个算法求交并比最坏情况下的复杂度是O($$n^2$$)，一般采用CUDA加速，时间复杂度从O($$n^2$$)变为O($$1$$)。

代码：

model/utils/nms/non_maximum_suppression.py

def _non_maximum_suppression_gpu(bbox, thresh, score=None, limit=None):



##### (b). 超参设置

后面需要使用c++对RoIs进行遍历，为了方便，需要把RoI存放到连续的GPU显存中：

```python
cp.ascontiguousarray(cp.asarray(roi)
```

设置非极大值抑制中的阈值 

```python
nms_thresh=0.7
```

设置非极大值抑制前RoIs的个数。训练时：从2万个anchor中选取概率较大的12000个anchor。

```python
n_train_pre_nms=12000
```

设置非极大值抑制后RoIs的个数。 训练时：从1.2万个anchor中，利用非极大值抑制，选出概率最大的2000个RoIs 。

```python
n_train_post_nms=2000
```



##### (c). RoIs排序

需要将RoIs根据RPN预测的前景概率从大到小排序：

```python
order = score.ravel().argsort()[::-1] # 将score拉伸并逆序（从高到低）排序 [1]
if n_pre_nms > 0:
    order = order[:n_pre_nms] # train时从20000中取前12000个rois，test取前6000个[1]
roi = roi[order, :]
```



##### (d). GPU资源分配

```python
threads_per_block = 64 # 设置每个block中线程的个数
```

```python
col_blocks = np.ceil(n_bbox / threads_per_block).astype(np.int32) # 12000 / 64 向上取整得 188 。设置block的个数
```

```python
blocks = (col_blocks, col_blocks, 1) # blocks可以是3维，这里只用两维,即一个grid中有col_blocks*col_blocks个block
```

```python
threads = (threads_per_block, 1, 1) # 一个block中有threads_per_block(64)个线程 [fan]
```

代码中的GPU资源分配如下图。分配的GPU资源可以看成一个grid(下图所有的block集合)，一个gird被分成了188\*188\*1个block(下图中的蓝色方框为一个block)，每个block中有64\*1\*1个线程(图中深蓝色的扭曲线条)，纵向上有188\*64 约等于 12000个线程，横向上有188\*64 约等于 12000个线程，代表着12000个框两两之间求交并比，每个线程对应了一次 两个框的交并比计算，采用并行计算，因此时间复杂度从O($$n^2$$)变为O($$1$$)。

![image-20191020214709631](image/35.png)

参考： https://blog.csdn.net/hujingshuang/article/details/53097222



##### (e). 使用CUDA计算两两框之间的交并比

以下代码使用CUDA 完成了 两两框之间计算IOU，并记录下与每个框 交并比超过阈值的框的位置信息，以便之后删除。这部分CUDA代码可在Python环境中被CuPy库调用。

```c++
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
//向上整除运算 [fan]
int const threadsPerBlock = sizeof(unsigned long long) * 8; //字节数*8 得到比特数

// Executed on the device(gpu),Callable from the device(gpu) only.   [fan] https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-function-specifier
__device__
inline float devIoU(float const *const bbox_a, float const *const bbox_b) { //devIoU计算两个框的交并比
  float top = max(bbox_a[0], bbox_b[0]); // 交集框的x1
  float bottom = min(bbox_a[2], bbox_b[2]); // 交集框的x2
  float left = max(bbox_a[1], bbox_b[1]); // 交集框的y1
  float right = min(bbox_a[3], bbox_b[3]); // 交集框的y2
  float height = max(bottom - top, 0.f); // 交集框的高
  float width = max(right - left, 0.f); // 交集框的宽
  float area_i = height * width; // 交集框的面积
  float area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]); // 框1的面积
  float area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]); // 框2的面积
  return area_i / (area_a + area_b - area_i); // 框1和框2的交并比
}

//extern "C" __global__   :   CUDA-level declaration of nms_kernel()  ， __global__ 代表是kernel函数 可并行，Executed on the device,Callable from the host, https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-function-specifier
extern "C"
__global__
void nms_kernel(const int n_bbox, const float thresh,
                const float *dev_bbox,
                unsigned long long *dev_mask) { // 这几个参数和封装函数kern的四个参数是一致的：cp.int32(n_bbox框的个数), cp.float32(thresh nms阈值), bbox(框的四个点), mask_dev(12000*12000) [fan]
  const int row_start = blockIdx.y; // 当前线程所在的block的行索引
  const int col_start = blockIdx.x; // 当前线程所在的block的列索引

  const int row_size =
        min(n_bbox - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_bbox - col_start * threadsPerBlock, threadsPerBlock); // 这里是每个block中线程的个数 ，与框的个数一一对应，有些block不需要那么多线程 [fan]

  __shared__ float block_bbox[threadsPerBlock * 4]; //一个block中线程可以共享这一个 block_bbox [fan]
  if (threadIdx.x < col_size) {
    //block_bbox是被比较的框，cuda grid中同一排的block_bbox应当是相同的[fan]
    block_bbox[threadIdx.x * 4 + 0] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0]; // 左上角的横坐标 c++访问数组的语法，尽管是二维数组，但是依旧可以 以一维数组的形式访问 [fan]
    block_bbox[threadIdx.x * 4 + 1] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1]; // 左上角的纵坐标
    block_bbox[threadIdx.x * 4 + 2] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2]; // 右下角的横坐标
    block_bbox[threadIdx.x * 4 + 3] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3]; // 右下角的纵坐标
  }
  __syncthreads(); // [fan] 块内线程同步 https://www.cnblogs.com/dwdxdy/p/3215136.html

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_bbox + cur_box_idx * 4;
    int i = 0; 
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {//这里解决的是 一个框与自己的交并比是1，这种情况会把自己抑制掉，又因为已排序，所以要从当前框下一个框开始。 [fan]
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_bbox + i * 4) >= thresh) { // 
        t |= 1ULL << i; //t记录的应该是当前线程对应的框 和 当前block对应的框的交并比 是否大于阈值[fan]
      }
    }
    const int col_blocks = DIVUP(n_bbox, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t; // t记录了交并比太高的框的位置信息。dev_mask的排列结构是:[框1_block1,框1_block2,,,,框1_block_last,,,,框last_block_last]
  }
}
```



##### (f). 删除交并比高于阈值的框

调用model/utils/nms/_nms_gpu_post.pyx ，根据mask标记交并比高的不要的框，之后删除。最后根据前景概率从大到小取前两千个框，NMS过程完成。



## 7.RoIHead/Fast R-CNN

### 7.1.RoIHead/Fast R-CNN网络结构

RPN只是给出了2000个候选框，RoI Head在给出的2000候选框之上继续进行分类和位置参数的回归。

![](image/30.jpg)



代码实现：

model/faster_rcnn_vgg16.py    class VGG16RoIHead(nn.Module):     



### 7.2.RoIs pooling

由于RoIs给出的2000个候选框，分别对应feature map不同大小的区域。首先利用`ProposalTargetCreator` 挑选出128个sample_rois, 然后使用了RoIPooling 将这些不同尺寸的区域全部pooling到同一个尺度（7×7）上。下图就是一个例子，对于feature map上两个不同尺度的RoI，经过RoIPooling之后，最后得到了3×3的feature map.

![](image/33.jpg)

RoI Pooling 是一种特殊的Pooling操作，给定一张图片的Feature map (512×H/16×W/16) ，和128个候选区域的座标（128×4），RoI Pooling将这些区域统一下采样到 （512×7×7），就得到了128×512×7×7的向量。可以看成是一个batch-size=128，通道数为512，7×7的feature map。

为什么要pooling成7×7的尺度？是为了能够共享权重。在之前讲过，除了用到VGG前几层的卷积之外，最后的全连接层也可以继续利用。当所有的RoIs都被pooling成（512×7×7）的feature map后，将它reshape 成一个一维的向量，就可以利用VGG-16预训练的权重，初始化前两层全连接。最后再接两个全连接层，分别是：

- FC 21 用来分类，预测RoIs属于哪个类别（20个类+背景）
- FC 84 用来回归位置（21个类，每个类都有4个位置参数）



代码实现：

model/roi_module.py    class RoI(Function):     def forward(self, x, rois):



### 7.3.RoIHead/Fast R-CNN训练

前面讲过，RPN经过NMS后会产生大约2000个RoIs，这2000个RoIs不是都拿去训练，而是利用`ProposalTargetCreator` 选择128个RoIs用以训练。选择的规则如下：

- RoIs和gt_bboxes 的IoU大于0.5的，选择一些（比如32个）
- 选择 RoIs和gt_bboxes的IoU小于等于0（或者0.1）的选择一些（比如 128-32=96个）作为负样本

为了便于训练，对选择出的128个RoIs，还对他们的`gt_roi_loc` 进行标准化处理（减去均值除以标准差）

对于分类问题,直接利用交叉熵损失.。而对于位置的回归损失，一样采用Smooth_L1Loss，只不过只对正样本计算损失。而且是只对正样本中的这个类别4个参数计算损失。举例来说:

- 一个RoI在经过FC 84后会输出一个84维的loc 向量. 如果这个RoI是负样本,则这84维向量不参与计算 Smooth_L1Loss
- 如果这个RoI是正样本,属于label K,那么它的第 K×4, K×4+1 ，K×4+2， K×4+3 这4个数参与计算损失，其余的不参与计算损失。

代码实现：

model/utils/creator_tool.py    class ProposalTargetCreator(object):	  def __call__()





## 8.RPN和RoIHead/Fast R-CNN 近似联合训练

近似联合训练。在这个解决方案中，RPN和Fast R-CNN网络在训练期间合并成一个网络。在我们的实验中，我们实验发现这个求解器产生了相当的结果，与交替训练相比，训练时间减少了大约25%−50%。这个求解器包含在我们发布的Python代码中。

(ii) Approximate joint training. In this solution, the RPN and Fast R-CNN networks are merged into one network during training as in Figure 2. In each SGD iteration, the forward pass generates region proposals which are treated just like fixed, pre-computed proposals when training a Fast R-CNN detector. The backward propagation takes place as usual, where for the shared layers the backward propagated signals from both the RPN loss and the Fast R-CNN loss are combined. This solution is easy to implement. But this solution ignores the derivative w.r.t. the proposal boxes’ coordinates that are also network responses, so is approximate. In our experiments, we have empirically found this solver produces close results, yet reduces the training time by about 25−50% comparing with alternating training. This solver is included in our released Python code.



代码中RPN和RoIHead/Fast R-CNN的损失 是简单的相加：

trainer.py

class FasterRCNNTrainer(nn.Module):

def forward(self, imgs, bboxes, labels, scale):





```
losses = [RPN_loc_loss, RPN_cls_loss, roi_loc_loss, roi_cls_loss]
losses = losses + [sum(losses)] # 四个损失函数求和追加到损失值列表尾 [fan]

losses.total_loss.backward()  #total_loss是加和后的损失 [fan]
```



## 9.测试推理

测试的时候对所有的RoIs（大概300个左右) 计算概率，并利用位置参数调整预测候选框的位置。然后再用一遍非极大值抑制（之前在RPN中 的`ProposalCreator`用过）。

注意：

- 在RPN的时候，已经对anchor做了一遍NMS，在RCNN测试的时候，还要再做一遍
- 在RPN的时候，已经对anchor的位置做了回归调整，在RCNN阶段还要对RoI再做一遍
- 在RPN阶段分类是二分类，而Fast RCNN阶段是21分类



代码实现：

model/faster_rcnn.py     class FasterRCNN(nn.Module):       def predict(self, imgs,sizes=None,visualize=False):



# 六.训练结果

超参设置：

```
voc_data_dir = './VOCdevkit/VOC2007'
min_size = 600  # image resize
max_size = 1000 # image resize
num_workers = 8
test_num_workers = 8

# sigma for l1_smooth_loss
rpn_sigma = 3.
roi_sigma = 1.

# param for optimizer
# 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
weight_decay = 0.0005
lr_decay = 0.1  # 1e-3 -> 1e-4
lr = 1e-3


# visualization
env = 'faster-rcnn'   #'faster-rcnn'  # visdom env
port = None #5002
plot_every = 40  # vis every N iter

# preset
data = 'voc'
pretrained_model = 'vgg16'

# training
epoch = 14


use_adam = False # Use Adam optimizer
use_chainer = False # try match everything as chainer
use_drop = False # use dropout in RoIHead
# debug
debug_file = '/tmp/debugf'

test_num = 10000
# model
load_path = None

caffe_pretrain = True # use caffe pretrained model instead of torchvision
# caffe_pretrain_path = '/students/julyedu_481243/fan/simple-faster-rcnn-pytorch-master/misc/checkpoints/vgg16_caffe.pth'
caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'
```



最后验证的结果：

lr:0.0001

mAP:0.6982598275398086 (与原论文中的0.699近乎相同)

loss:{'rpn_loc_loss': 0.03477665711630759, 'rpn_cls_loss': 0.03971287968107227, 'roi_loc_loss': 0.1241183091533306, 'roi_cls_loss': 0.08903295600408019, 'total_loss': 0.28764080180097296}