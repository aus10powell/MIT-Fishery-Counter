# MIT Fishery Counter
![Red Herring](./pics/RedHerring.png)

### Contributors
* *Robert Vincent (Research Advisor)*
* *Austin Powell (Contributor)*
* *Lydia Zuehsow (Contributor)*
* *Blaine Gilbreth (Contributor)*

## Overview
Fisheries populations have a large impact on the U.S. economy. Each year the U.S. fishing industry contributes 90 billion dollars and 1.5 million jobs to the U.S. economy. Each species may serve as a predator or prey for another. In this regard, fisheries populations are interconnected and dependent. While humans may depend on these populations as a source of sustenance (food, goods, etc.), humans can also negatively impact population growth. Barriers to migration, pollution, overfishing, and other forms of human-interference may impact spawning patterns of fisheries species. In 2014, 17% of U.S. fisheries were classified as overfished. Therefore, it is necessary to monitor these fisheries populations to determine when policy must be changed in efforts to maintain healthy oceans.

Many groups, including NOAA Fisheries, state agencies, as well as regional fisheries councils and local municipalities, deploy camera and video equipment to monitor fisheries populations. Large amounts of video and photographic data are gathered at timed intervals. However, not all photos contain aquatic life. Currently, employees at these agencies among others are responsible for manually annotating the gathered videos and photos; this means they identify and count the relevant aquatic specimens in the data. Not only is this an inefficient use of time and resources, but also it can lead to inaccurate results due to human error. NOAA Fisheries Management can make a significant improvement in time and resource use through automation of the annotation process.

## Methods
A combination of optical flow and background removal is used in this analysis. These techniques are useful in extracting information from video, especially when movement is involved. To begin, OpenCV background removal is used to isolate moving parts of the image, such as the fish or variations in the water motion. A mask of these areas of the image is applied, so that the image is black and white. Morphological transformations are then applied to an image with the background removed, so that random isolated points are not included and the part representing the fish is expanded and connected. Contours describing regions where the mask is present are found and boxed. These bounding boxes are then analyzed for their size and location.

Separately, but on the same video feed, optical flow is used to find key points in the image and analyze their movement over time to determine fish directionality. In particular, Lucas-Canade optical flow is used, which looks at the movement of a few selected points. Points that are within the above-described bounding boxes are included in directionality analysis. This is done by averaging the movement of key points within a bounding box to determine a probable direction of the fish.

Through these two methods, a counter is implemented such that fish are tracked across the screen and added to the counter if they are moving right to left. There are various parameters that can be used modify the tracker for different input parameters, such as the number of frames it takes into account before a fish hits the center and whether we include fish where it does not find directionality data.

## Objectives:
1. Identify fish as River Herring and Not River Herring (or by individual species)
2. Count River Herring Only
3. Count fish passing by in only one direction
4. Date and time stamps
5. Provide summary statistics output tables and figures
6. Provide an estimate of identification and count accuracy
7. Develop software that can be given to fisheries managers so that they can run the software on their own video and have the computer count and total fish for their spawning run season
8. Develop real-time monitoring and counting in the field with weekly downloads
9. Develop a user interface for non-computer programmers
10. Include other monitoring sensor and output such as temperature, current speed, light, and measurements of fish length and biomass would be desirable as well.

## Setup
Current setup is a gated entry into the resorvoir in order to allow cameras to capture all of the content in cages.

### Cameras and gates
<p float="left">
  <img src="./pics/Camera_Set_Up_20220425_rv1.jpeg" width="300" height="300"/>
  <img src="./pics/Camera_Set_Up_20220425_rv2.jpeg" width="300" height="300"/> 
  <img src="./pics/Camera_Set_Up_20220425_rv3.jpeg" width="300" height="300" /> 
  <img src="./pics/Camera_Set_Up_20220425_rv4.jpeg" width="300" height="300"/> 
</p> 

### Herring Example Photos
<p float="left">
  <img src="./pics/HerringSylviaPlace.png" width="300" height="300"/> 
    <img src="./pics/HerringScreenCap.png" width="300" height="300"/> 
</p>

## Documentation
* [Instruction Manual](/documentation/Instruction%20Manual.pdf)

### Dropbox
Currently linking to Dropbox folders for large files
* [Original Project Files](https://www.dropbox.com/sh/26y1pqukooepsmr/AADXGlkRWTFrKl9GwN1SpDRUa?dl=0)
* [2017 Software](https://www.dropbox.com/sh/26y1pqukooepsmr/AAD5wAO3EgLeyT7ENfvqhgIIa/2017%20Software?dl=0&subfolder_nav_tracking=1)

### Whitepapers
* Read more [AMS Presentation](/documentation/AMS%20Presentation.pdf)

## Future-Work:
* Implement automation of fish counting from images that removes manual process of running on individual's machines
* Continue applying image recognition to herring:
	* Of interest to NOAA Fisheries, state agencies, as well as regional fisheries councils and local municipalities.
	* Image recognition is a novel approach
* Develop graphical user interface for end-users
* Test other image recognition algorithms, such as Faster R-CNN and Mask R-CNN