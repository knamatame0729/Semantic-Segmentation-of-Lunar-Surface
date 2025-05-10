# ORBSLAM3 Semantic Mapping for RAS598 Final Project at Arizona State University
## Overveiw
**This repository is for a project in RAS 598 calss at Arizona State University**
- Midterm Project  
I have done implementing an encder-decoder neural network based on U-net arhitecture with VGG16 as an encoder, specifically developed for rover-based lunar terrain segmentation.  
[Midterm Report PDF](Midterm_Report.pdf)
- Final Project  
I have done integrating semantic segmentation and ORB SLAM3 to overcome cons of both semantic segmentation and 3D point cloud provided by ORB SLAM3.  
[Final Report PDF](FinalProject_Report.pdf)

## Demo Video  

<video controls src="FinalProject_video.mp4" title="Title"></video>

## Introduction
**ORBSLAM_Semantic_Mapping** is based on [ros2_orb_slam3](https://github.com/Mechazo11/ros2_orb_slam3). ORB SLAM3 is a great SLAM method that has been applied robot application. However, this method can not provide semantic information in environmental mapping. In this project, I present a method to build a 3D semantic map, which utilize both 2D semantic images from semantic segmentation model (U-Net with a VGG16 as encoder) and 3D ponit cloud map from ORB SLAM3.  

![Image](https://github.com/user-attachments/assets/a7596ac9-7593-4521-acb6-af31450a2db2) 

## Challenge and Future Plan



