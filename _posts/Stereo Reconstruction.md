---
layout:     post
title:      Growth measurement of small plants in 3D by Using Stereo Reconstruction
subtitle:   
date:       2015
author:     Mary Li
header-img: img/
catalog: true
tags:
    - Stereo reconstruction, 
    - 3D mesh
    - Plant Phenotyping
    - Point cloud processing
---

## Introduction

High throughput phenotyping platforms, when combined with 3D image analysis, enable researchers to inves-
tigate complex functional traits related to plant structure, including responses to external and internal signals
or perturbations. In this research, we present a computational workflow for analysing the growth of Arabidopsis thaliana rosettes over time using stereo reconstruction. 

Main achievements:
i) the system is able to construct realistic point clouds and meshes of the scene,
ii) the processing pipeline is computationally efficient,
iii) it allows stereo reconstruction and segmentation of individual plants in trays of 20 plants each for high throughput analysis.
The overall approach proved useful in quantifying morphometric parameters in 3D for a set of Arabidopsis
accessions and relating plant structure to plant function.

![](/img/icon.jpg)
>Keywords：Stereo reconstruction, 3D mesh,  Plant Phenotyping, Point cloud filtering

## Method

The platform,enclosed 3 types of cameras mounted overhead in three stations:a stereo RGB camera station,a FLIR infrared imaging station, and a pulse modulated chlorophyll fluorescence imaging station. A integrated processing pipeline has been developed to process all three data input (here I only include published data on RGB imaging). 


### stereo reconstruction 

The stereo reconstruction module is based on the semi-global matching method. After images were rectified, pairs of stereo images were matched using semi-global matching. The disparity map was generated along with the 3D point cloud for the whole tray.

![](https://cdn.mos.cms.futurecdn.net/RdxhPVv8fAyM6oHsRgF6dH-650-80.png)

###  Plant Segmentation

After generating the initial point cloud, we developed a plant segmentation module to retain only the 3D points
of interest (i.e., the plants) while removing the background points. This step involves two processes: green
area segmentation and noise filtering. Following this step, each plant was individually extracted by point cloud clustering.

![](https://images.ifanr.cn/wp-content/uploads/2018/06/WWDC-10.jpg)


### Surface Reconstruction and Mesh Generation

The next step was to generate surface meshes based on individual plant point clouds. The mesh generation
module consists of two parts: a combination of the Greedy Projection and a hole filling algorithm followed by
a Poisson Reconstruction.

![](https://images.ifanr.cn/wp-content/uploads/2018/06/WWDC-13.jpg)

![](https://images.ifanr.cn/wp-content/uploads/2018/06/WWDC-14.jpg)

### Experiment result

After generating individual mesh, the surface area was calculated by summing the areas of each
triangle in the mesh. The result of a sample tray for captured plants is shown below.

![](https://cdn.sspai.com/2018-06-04-Artboard.jpg?imageView2/2/w/1120/q/90/interlace/1/ignore-error/1)



### 

### Reference

- 
 

