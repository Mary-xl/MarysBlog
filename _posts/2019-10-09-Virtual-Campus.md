---
layout:     post
title:      Virtual Campus
date:       2019-10-09
author:     Mary Li
header-img: img/2019-10-09-bg.jpeg
catalog: true
tags:
    - 3D mapping and reconstruction
    - VR/AR
    - image processing
---

_For my bachelor’s degree thesis in 2009, I developed a virtual campus system based on the International School of Software in Wuhan University. Although some of the technologies used in the system seem  naive and outdated today, it opened the door to the world of computer vision and 3D reconstruction for me. Since then I chose to pursue my career in this field  that I felt passionate about. This thesis received “Best Undergraduate Thesis”, the second prize in Hubei Province, China, 2009._ 

<iframe width="560" height="315" src="https://www.youtube.com/embed/vkeJeaIkTEQ" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

## Introduction

This system was built with VRML(Virtual Reality Modeling Language), a light weight 3D modelling language that was no longer active today. But its intention to allow users to visualise and interact with VR applications the same way as web browsing based on HTML is still fascinating. 

## Methods
Four main steps were taken to build the virtual campus: 3D modelling, multi-scene integration, user-virtual environment interaction and browser/Server deploy and publishing.

### 3D modelling
3D modelling was based on the surveyed geometry information of  the campus, including buildings, roads, trees etc. Then images were taken and rendered on the these 3D models in 3DsMax to constitute basic scenes. 

![2019-10-09-11.png](https://i.loli.net/2019/10/10/lq5dtJrLhRXWjzP.png)
![2019-10-09-5.png](https://i.loli.net/2019/10/10/Bl3ciDT6jALqY5r.png)
![2019-10-09-2.png](https://i.loli.net/2019/10/10/umgRcsdVoGLZhHS.png)
![2019-10-09-4.png](https://i.loli.net/2019/10/10/yXWJi4IDx16nRVM.png)

For natural landmarks such as trees and vegetation, a trade-off between level of details and rendering speed needs to be considered. 
![2019-10-09-13.png](https://i.loli.net/2019/10/10/ua5XRnmwjciZzWh.png)

### Multi-scene integration

Next each of the basic scenes were integrated in Cosmo Worlds to form the entire virtual campus. A ortho-photo of the campus was used as the geo-reference map. Then “inline” node in VRML was used to include different models into the same environment, including localisation and scaling. 
Using Cosmo Worlds:
![2019-10-09-9.png](https://i.loli.net/2019/10/10/7PgyOnvpzkMdaGf.png)

Localisation and scaling:
![2019-10-09-6.png](https://i.loli.net/2019/10/10/pqbEZSj2dwraylM.png)
![2019-10-09-7.png](https://i.loli.net/2019/10/10/YLAni7PsUHQShZE.png)
![2019-10-09-8.png](https://i.loli.net/2019/10/10/rqs2DUe54ZOCQoh.png)

### User-virtual environment interaction

Interaction was realised via different sensors defined in the VRML: TimeSensor, TouchSensor, ProximitySensor, Collision and so forth. They enabled the virtual environment to “sensor” the user’s view, proximity, and collision and “react” accordingly. For instance, when the user’s avatar is close to the woods, user can hear bird tweeting. Collision sensor makes sure that the user won’t fall into the ground/buildings.  It also enables user to change from one scene to another , such as outdoor and indoor.
![2019-10-09-10.png](https://i.loli.net/2019/10/10/PVOHXTpKn9eSLqw.png)

### Browser/Server deploy and publishing

Finally the virtual campus was deployed on the server (tomcat5.5) and published based on a Server-Browser structure.
