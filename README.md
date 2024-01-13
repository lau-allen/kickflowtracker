# KickFlowTracker
The goal of KickFlowTracker is to utilize computer vision and machine learning techniques for ball detection, extraction, and speed estimation in football. This is accomplished using the following pipeline:
* video processing - extract frames from video input and preprocess them 
* ball tracking (non-deep learning approach) - utilize Hough Transform (CHT) for ball detection and Channel and Spatial Reliability of Discriminative Correlation Filter (CSRT) for ball tracking 
* ball tracking (deep learning approach) - YOLO 
* speed estimation - Farneback Optical Flow 

## Major Files Breakdown 
* [kickflowtracker.py](https://github.com/lau-allen/kickflowtracker/blob/main/kickflowtracker.py) - Entry point for the KickFlowTracker app. 
* [camera_utils.py](https://github.com/lau-allen/kickflowtracker/blob/main/utils/camera_utils.py) - Logic for camera calibration, detecting circles via CHT, and other camera-related utils. 
* [motion_utils.py](https://github.com/lau-allen/kickflowtracker/blob/main/utils/motion_utils.py) - Logic for optical flow speed estimation. 
* [kickflowtracker_report.pdf](https://github.com/lau-allen/kickflowtracker/blob/main/kickflowtracker_report.pdf) - Report outlining the problem, objective, related work, method, results, and conclusions for kickflowtracker. 
* [kickflowtracker_presentation.pdf](https://github.com/lau-allen/kickflowtracker/blob/main/kickflowtracker_presentation.pdf) - Presentation slide deck. 