# KickFlowTracker
The goal of KickFlowTracker is to utilize computer vision and machine learning techniques for ball detection, extraction, and speed estimation in football. This is accomplished using the following pipeline:
* video processing - extract frames from video input and preprocess them 
* ball tracking (non-deep learning approach) - utilize Hough Transform (CHT) for ball detection and Channel and Spatial Reliability of Discriminative Correlation Filter (CSRT) for ball tracking 
* ball tracking (deep learning approach) - YOLO 
* speed estimation - Farneback Optical Flow 

## Major Files Breakdown 
* [kickflowtracker.py]()