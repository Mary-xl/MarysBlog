---
layout:     post
title:      Vision-based Positioning & Navigation  with the use of Geo-referenced 3D Maps
date:       2019-10-07
author:     Mary Li
header-img: img/2019-10-07_bg.png
catalog: true
tags:
    - 3D mapping 
    - visiual navigation
    - feature extraction
    - image matching
    - photogrammetric space resection
---

## Introduction

Ubiquitous positioning is considered to be a highly demanding application for today’s Location-Based Services (LBS). While satellite-based navigation has achieved great advances in the past few decades, positioning and navigation in indoor and deep urban areas has remained a challenging topic. In this research, we present a hybrid image-based positioning system that is intended to provide seamless position solution in 6 degree of freedom (6DoF) for location-based services in both outdoor and indoor environments. It mainly uses visual input to match with geo-referenced images for image-based positioning resolution, and also takes advantage of multiple sensors onboard, including built-in GPS receiver and digital compass to assist visual methods. Experiments demonstrate that such system can largely improve the position accuracy for areas where GPS signal is degraded (such as in urban canyons), and it also provides excellent position accuracy for indoor environments.

>Keywords：3D mapping, feature extraction, visiual navigation, image matching, photogrammetric space resection

## Methodology

Positioning is an essential component in navigation systems. It mainly functions in two different ways: absolute self-localization (e.g. GPS) and dead-reckoning (e.g. inertial navigation system). In this paper, we propose a hybrid image-based navigation system that is capable of self-localization in both outdoor and indoor environments. It requires a mapping process where images of the navigation environment are captured and geo-referenced. The main improvements in this work are to geo-reference image feature points and use these features as 3D natural landmarks for positioning and navigation. By matching the real time query image with pre-stored geo-referenced images, the 3D landmarks represented by feature points are recognized and their geo-information are transferred from reference image to query image through these common feature points. Final positioning is based on photogrammetric space resection.

![2019-10-07_1.png](https://i.loli.net/2019/10/08/7z2la4LkQoWfTi3.png)
### Image Geo-Referencing and Mapping 

To produce geo-referenced 3D maps for both indoor and outdoor positioning, ground control points have been set up and surveyed (mm level RTK GPS) while and the images of the navigational environment are collected. Then feature points (eg. SIFT, Harris Corners) are extracted from these images.
![2019-10-07_4.jpg](https://i.loli.net/2019/10/08/swlbWZGnxkNC5LT.jpg)

Then feature based image matching is performed between images with overlapped areas to produce tie points. These tie feature points are then geo-referenced through triangulation and non-linear optimization solution of bundle adjustment. The geometric accuracy of the map depends on the accuracy of geo-referencing. The overall flowchart for mapping is shown in the following figure.
![2019-10-07_2.png](https://i.loli.net/2019/10/08/U8kNRjdeEvzYlap.png)
###  Image-based Positioning

During the navigation process, query images are taken wherever self-localization is needed. Although the method varies for indoors and outdoors, the processes for the two environments are based on the same principles: by performing the image matching between real time query image and reference images, 3D coordinates are transferred through common features from reference images to the query image.  By obtaining the 3D coordinates of feature points and their 2D coordinates on the query image, camera position and orientation in 6 degree of freedom can be determined through photogrammetric space resection. 

![2019-10-07_3.png](https://i.loli.net/2019/10/08/fjPziybqTA8ZxH2.png))


## Outdoor Positioning

In urban canyons or indoor areas, GPS positioning accuracy can be degraded because the signal may suffer from blockage, multi-path effects, etc. For single point positioning (SPP) used on people’s mobile devices, the accuracy can be 10s meters or worse. Therefore, image-based methods are used to mitigate the deficiency. However, if solely replace GPS with image-based methods, retrieving images from a large image database that covers the whole navigation route will be time consuming and the computation load is not affordable for mobile devices. Therefore, we propose a multi-step solution: 

### Using GPS to Localize Image Space

Whenever a query image is taken with its GPS position tagged, the initial position is given by the GPS tag and the initial orientation is given by the digital compass onboard.  A search radius will be generated at current GPS tagged position and all relevant geo-referenced map images will be loaded.

![2019-10-07_5.png](https://i.loli.net/2019/10/08/Xcelng9VEwYIFTP.png)
![2019-10-07_6.png](https://i.loli.net/2019/10/08/zakGQwUA6JKLCXq.png)

### Image Retrieval using SIFT-based Voting Strategy

Given the query image:
![2019-10-07_7.png](https://i.loli.net/2019/10/08/nTcZDVpwSMxv5Hi.png)

 a voting strategy is used to find reference images corresponding to the query view among the localized image space. SIFT matching is performed between the query image and the reference feature database to find corresponding reference images. A K-NN search function is used to find the k nearest neighbours from feature database for the feature points in the query image. Each correspondence found adds one vote to the reference image it belongs to. The reference images with greater numbers of votes obviously have higher chance of containing common scene with query image. Therefore, by ranking in descending order of the number of votes, the best candidate (5 in our case) reference images are identified. 

![2019-10-07_8.png](https://i.loli.net/2019/10/08/48wuhzyrFe1AGaU.png)



### Outdoor Positioning Result

After getting the best candidate reference images, image-matching is performed to find common feature points and their 3D coordinates are transferred from the map to the query image for positiong calculation. It's noted that measurements from GPS and digital compass are used as initial estimate for a final least square based positioning calculation in outdoor environments. 

![2019-10-07_9.png](https://i.loli.net/2019/10/08/xFyNlLI25H1zMUG.png)
![2019-10-07-10.png](https://i.loli.net/2019/10/08/K4jvX7iWBfhPNkG.png)

It can be seen that the GPS measurements in the urban environment are poor, around 20m in our experiments. By using the proposed method, the accuracy has been improved to around 10m revealed in the test.

## Indoor Positioning and Result

Since the target building has been identified in Section 3, when a user walks into the building, the geo-referenced images of its indoor environment are loaded. Then real time images are taken. Image matching based on SIFT is carried out between the query image and the geo-referenced images for position resolution.

<video width="320" height="258" controls>
<source src="https://youtu.be/94zQflwlEio">
</video>

{% include youtube.html id=" https://youtu.be/94zQflwlEio" %}

![2019-10-07-11.png](https://i.loli.net/2019/10/08/mdO3vQC5LpzEgVT.png)
![2019-10-07-12.png](https://i.loli.net/2019/10/08/3dyc2aB5JFYZCzU.png)

From the video, a total 83 epochs (frames) were generated & calculated, 20 epochs failed to determine the camera position, which is failure rate at 24.1%. It is observed that the failed or in accurate results all come from insufficient feature points due to unevenly distributed texture. A possible solution is to set up artificial marks in areas where no texture can be found, such as blank walls.

## Conclusions

In this research, we have presented a comprehensive system that adopts a hybrid image-based method with combined use of onboard sensors (GPS, camera and digital compass) to achieve a seamless positioning solution for both indoor and outdoor environments. The main contribution are:
- A combined use of photogrammetric methods and computer vision algorithms;
- The use of geo-referenced images as 3D maps for image-based positioning;
- The adoption of multiple sensors to assist the position resolution.

Experiments have demonstrated that such a system can largely improve the position accuracy for areas where GPS signal is degraded (such as in urban canyons). The system also provides excellent position accuracy (20 cm) for indoor environments.
The nature of such system has also been studied. The final position accuracy is mainly determined by the geometry (number & distribution) of the identified geo-referenced features. Therefore, the geo-referenced 3D feature density of the reference images, the quality of image matching and most importantly the scene of the query image become the essential elements of the solution. It therefore has limitations in environments with shortage of texture or poor lighting conditions. Moreover, in complex indoor environments where initial estimates are hard to achieve, other positioning method needs to be introduced to assist vision-based methods. 

## Related Publications

- Li, X., Wang, J., & Li, T. 2013. Seamless Positioning and Navigation by Using Geo-Referenced
Images and Multi-Sensor Data. Sensors, 13(7), 9047-9069.
- Li X., Wang J., Knight N., & W. Ding 2011. Vision-based Positioning with a Single Camera and 3D
Maps: Accuracy and Reliability Analysis. Journal of Global Positioning Systems.
- Li X., Wang, J. 2014. Image Matching Techniques for Vision-based Indoor Navigation Systems: A 3D
Map Based Approach. Journal of Location Based Services, 8(1), 3-17.

- Li X., Wang J., 2012. Image Matching Techniques for Vision-based Indoor Navigation Systems:
Performance Analysis for 3D Map Based Approach. Proceedings of the 2012 International
Conference on Indoor Positioning and Indoor Navigation (IPIN 2012), Sydney, Australia, 13-15th,November.
- Li X., Wang J., 2012. Multi-image Matching for 3D Mapping in Vision-based Navigation Applications.Proceedings of the Ubiquitous Positioning Indoor Navigation and Location Based Service (UPINLBS),
Finland, Oct. 2012.
- Li X., Wang J., 2012. Evaluating photogrammetric approach of image-based positioning. Proceedings
of the XXII Congress of International Society for Photogrammetry & Remote Sensing (ISPRS2012),
Melbourne, August 2012
- Li X., Wang J., Li R., Ding W., 2011. Image-based positioning with the use of geo-referenced SIFT
features. Proceedings of the International Symposium on GPS/GNSS (IGNSS 2011), Sydney,
Australia, November 2011.
- Li X., Wang J., Liu W. & R Li., 2013. Geo-referenced 3D Maps: Concept and Experiments. The 8th
International Symposium on Mobile Mapping Technology, Tainan, Taiwan,1-3 May.
- Li X., Wang J., Yang L., 2011. Outlier Detection for Indoor Vision-Based Navigation Applications.
Proceedings of the 24th International Technical Meeting of The Satellite Division of the Institute of
Navigation (ION GNSS 2011), Portland, OR, September 2011.
- Li X., Wang J., Knight N., Olesk A., Ding W., 2010. Indoor positioning within a single camera and 3D
maps. IEEE Proceedings of Ubiquitous Positioning Indoor Navigation and Location Based Service
(UPINLBS), 2010,Oct.

