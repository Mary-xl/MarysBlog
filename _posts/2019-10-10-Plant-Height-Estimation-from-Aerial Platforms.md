---
layout:     post
title:      A point cloud based approach for plant height estimation from aerial platforms
date:       2019-10-10
author:     Mary Li
header-img: img/2019-10-10-bg.png
catalog: true
tags:
    - 3D mapping and reconstruction
    - point cloud processing
    - Agriculture-Plant Phenotyping
    - image processing
---
_In 2018-2019,I led a project to develop an aerial data processing pipeline that aims to extract various plant traits by minimum number of flights. 
The software should be able to process data collected by both manned (helicopter) and unmanned (UAV) vehicle and segment single plant/plot from large scale 
agriculture field for trait analysis. I include here the published part of the research. Please cite the paper at bottom for any information from this article._ 


## Introduction

With the development of computer vision technologies, using images acquired by aerial platforms to measure large scale agricultural fields has been increasingly studied. In order to provide a more time efficient, light weight and low cost solution, in this paper we present a highly automated processing pipeline that performs plant height estimation based on a dense point cloud generated from aerial RGB images, requiring only a single flight. A previously acquired terrain model is not required as input. The process extracts a segmented plant
layer and bare ground layer. Ground height estimation achieves sub 10cm accuracy. High throughput plant height estimation has been performed and results are compared with LiDAR based measurements.
The key innovation is to estimate
ground height underneath the target plant area using an inverse
distance weighting from visible points of the neighbouring
ground in form of gridded ground controids. It not only
enables plant height calculation at plot level, but can also
be applied to various shapes of canopies for different types
of plants. The only manual input required is the marking of
Ground Control Points (GCP) and contour labeling(annotation) of the
Regions of Interest (ROI) in the plot detection process.

## METHODOLOGY

The proposed processing pipeline consists of multiple steps,
as shown in Figure 1: 1) image acquisition and filtering
; 2) plot automated detection from the ortho-mosaics 3) generation of dense 3D point cloud; 4) point cloud
segmentation; 5) ground height estimation; 6) plant height
estimation. In this paper we focused on 3-6 and deep learning based plot detection will discussed separately in another paper.

### Image Acquisition and 3D reconstruction

An integrated unit that can be mounted on both manned or unmaned aerial platforms was designed for this research. A high resolution camera (CanonEOS6D) was integrated in the unit for 3D reconstruction. In this experiment images were taken from a helicopter with 90% frontal overlap and 70% side overlap on average. The flight height was approximately 90 meters above ground with a speed of travel at 25-30 knots. A total of 9 evenly distributed GCPs at centimeter level accuracy were used together with 539 images, covering
an area of 0.168km^2. A 1.89cm Ground Sampling Distance (GSD) has been achieved. 

Full 3D reconstruction was done by Pix4Dmapper Pro, which is a commercial software package that uses structure from motion ([SFM](http://marysfishingpool.tk///2019/10/12/SFM/)) techniques. It corrects camera distortion, refines location using ground control points (GCPs), accounts for camera poses and generates 3D maps in the form of ortho-mosaic and 3D point cloud. We mainly use the 3D point cloud for further analysis:
![field.gif](https://i.loli.net/2019/10/11/KtXDkHJ4Ijl9orb.gif)

### Point cloud segmentation

![Kqkj1O.gif](https://s2.ax1x.com/2019/11/02/Kqkj1O.gif)
<center> Plant Layer Segmentation </center>
The point cloud segmentation process consists of three parts: plant layer segmentation, ground layer segmentation and ground surface smoothing. More specifically, we first segment the plant layer based on colour indices. 
<br>
[![KboZOH.md.png](https://s2.ax1x.com/2019/11/02/KboZOH.md.png)](https://imgchr.com/i/KboZOH)
<center>Plant Layer</center>
The remaining points are the soil ground as well as other objects in the field, such as buildings, vehicle and dark or less green vegetation. In order to extract the bare ground, a Progressive Morphological Filter
[1] was used to filter out these objects. The filter uses classical morphological opening with a gradually increasing window size and elevation differences to remove non-ground
objects while retaining the terrain slope changes. This method has previously been used in
filtering LiDAR point clouds and DEM generation, and it is increasingly being adopted in Photogrammetric research. 
<br>
[![KboMkt.md.png](https://s2.ax1x.com/2019/11/02/KboMkt.md.png)](https://imgchr.com/i/KboMkt)
<center> Ground Layer </center>
[![Kbodkq.md.png](https://s2.ax1x.com/2019/11/02/Kbodkq.md.png)](https://imgchr.com/i/Kbodkq)
<center> Object Layer </center>
The resulting point cloud, the bare ground, is still noisy. Moving least squares (MLS) surface reconstruction [2] is used to remove noise and build a smooth ground surface from the point cloud. Compared with other noise filtering and surface reconstruction methods, MLS is more powerful in dealing with intrinsic errors such as data irregularities due to small object measurements. The idea is to use an analytical function to approximate surfaces based on weighted local data points. It is being increasingly adopted as the standard definition of point set surfaces, and various algorithms in this class have been proposed [3]. Afterwards, colour information is further removed to reduce data size.
![KbTI5q.png](https://s2.ax1x.com/2019/11/02/KbTI5q.png)

### Grid based ground height estimation

Once a ground surface has been obtained, the next step is to calculate the ground height. At this stage, the plants inside each plot have been removed, leaving only holes. We then grid the ground surface and calculate ground heights based on point clouds within each grid.
A 2D bounding box in the XY plane of the ground surface is used to divide the ground into regular rectangle grids. Within each grid, a sub-point cloud is vertically extracted and a centroid of the point cloud is calculated for all axes (X,Y,Z). This centroid is called the ground centroid (GC). The centroid Z value is used as the height for this grid. For this
particular dataset, a 50×50 rectangular grid was used, resulting in 11.2m x 7.45m size single grid unit for height calculation (see Figure 9). Therefore, ground height has been calculated
at 50 × 50 resolution with each grid covering approximately 80 m^2 . If a grid has no data points, no centroid is generated.
![Screenshot from 2019-10-11 16-03-18.png](https://i.loli.net/2019/10/11/aZOtIMqhcGEFXkP.png)

### Plant height estimation 

Plant height estimation consists of two steps: 1) region of interest (ROI) extraction; 2) height estimation for each ROI.

####  region of interest (ROI) extraction
The ROI is extracted by employ state of the art FasterRCNN, where contours/bounding boxes of plots and trees are detected and located on top of the ortho-mosaic (this part hasn't been published).
Their 2D bounding boxes are then used for 3D point cloud segmentation. 
![polygon_roi.gif](https://i.loli.net/2019/10/11/MlwS6yVoriHL7vI.gif)

#### Plant height estimation using inverse distance weighting
After each ROI point cloud has been extracted, its height is
calculated by subtracting the local ground height from each
point height. Then a histogram of height values is generated
for each ROI and the average value is used as the overall
height for each ROI. The key element is to calculate the local
ground height underneath each ROI.

By using the ground centroids calculated as shown in
section "Grid based ground height estimation", we are able to obtain the ground height at various locations. A 2D centroid for each ROI is calculated based on its 2D contour and used as its position. Then k-
nearest neighbour (k-NN) search is used to find its k nearest ground centroids within 2D space. Based on the 2D Euclidean
distances between the ground centroids and the ROI centroid,
an inverse distance weighting (IDW) algorithm is used to
estimate the height at the ROI centroid location. That height
is used as the local ground height for the ROI. The process is shown in Algorithm 1 and illustrated in the following figure:
![Screenshot from 2019-10-11 16-09-42.png](https://i.loli.net/2019/10/11/GHi3pfBThUw1OMK.png)
![Screenshot from 2019-10-11 23-06-20.png](https://i.loli.net/2019/10/11/T1VvFOdQMkHf5i7.png)


## Experiments and Discussion

We first validate our ground height estimation accuracy by comparing it with 31 evenly distributed check points.The root mean square error (RMSE) between the ground
height measured by RTK GPS and by our method is 8.2cm.
The agreement between our method and the reference data is
shown in Figure 13.
![Screenshot from 2019-10-11 21-47-13.png](https://i.loli.net/2019/10/11/HqCs6prljiORQMK.png)

Data from four experiments of total 784 plots have been
used to test the height estimation pipeline. All 784 ROI have been extracted from the plant layer so as generating 784 point clouds. The height for each ROI has
been estimated and compared with Lidar data collected 2 days before the flight. 
![Screenshot from 2019-10-11 22-38-34.png](https://i.loli.net/2019/10/11/lquMDEVf35UtnCS.png)

A consistent offset between aerial measurement and Lidar can be observed. Interestingly, the same phenomenon has been reported in other research too. In [4], compared with
ground LiDAR, a systematic under-estimation is observed for structure from motion techniques (aerial point cloud derived plant height) . The authors believe their result agrees with previous studies [5] where it is found that structure from motion from aerial imagery lacks the ability to reconstruct accurately the top of the canopy, due to the reason that
aerial RGB imagery has a coarser spatial resolution and limited penetration capacity compared with ground LiDAR. Future research will be focused mitigating these two
shortcomings with other sensors and technologies.

## Development
I personally contributed to over 80% of system design and development (the whole backend pipeline), including the main innovation of ground height estimation of invisible areas and various point cloud processing procedures for layer segmentation and height estimation.
Programming aspect: C++, Python, PCL library, OpenCV, Pytorch



## Reference
[1] K. Zhang, S.-C. Chen, D. Whitman, M.-L. Shyu, J. Yan, and C. Zhang,
“A progressive morphological filter for removing nonground measure-
ments from airborne LIDAR data,” IEEE Transactions on Geoscience
and Remote Sensing, vol. 41, no. 4, pp. 872–882, apr 2003.

[2] M. Alexa, J. Behr, D. Cohen-Or, S. Fleishman, D. Levin, and C. Silva,
“Point set surfaces,” in Proceedings Visualization, 2001. VIS’01. IEEE,
2001.

[3] Z.-Q. Cheng, Y.-Z. Wang, B. Li, K. Xu, G. Dang, and S.-Y. Jin, “A
survey of methods for moving least squares surfaces,” 2008.

[4] S. Madec, F. Baret, B. de Solan, S. Thomas, D. Dutartre, S. Jezequel,
M. Hemmerlé, G. Colombeau, and A. Comar, “High-throughput pheno-
typing of plant height: Comparing unmanned aerial vehicles and ground
LiDAR estimates,” Frontiers in Plant Science, vol. 8, nov 2017.

[5] G. J. Grenzdörffer, “Crop height determination with UAS point clouds,”
ISPRS - International Archives of the Photogrammetry, Remote Sensing
and Spatial Information Sciences, vol. XL-1, pp. 135–140, nov 2014.

## Related Publication
Li X., Bull G., Coe R., Eamkulworapong S., Scarrow J., Sirault X., Salim M., Schaefer M., 2019.
High-Throughput Plant Height Estimation from RGB Images Acquired with Aerial Platforms: a 3D
Point Cloud based Approach. Accepted by the International Conference on Digital Image Computing:Techniques and Applications (DICTA 2019). Perth, Australia, 02 Dec - 04 Dec 201


