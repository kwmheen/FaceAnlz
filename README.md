# FaceAnlz - face detecting library
Simple Face Detecting library  with mediapipe, google open source
***

## Table Of Contents
* [Introduction](##Introduction)
* [Technologies](##Technologies)
* [Setup](##Setup)
* [Document](##Document)
* [Example Code](##Example)
* [Source](##Source)

## Introduction
> This is Simple Face Detecting Library with MediaPipe, google open source.
Extracting Face Detection Box, Face Mesh (Face Polygon) From Video or Static Image.
Can Adjust Perception Reliability, Maximum Face Number.

## Technologies
Project is created with:
* Python Version: 3.10.4
* MediaPipe Version: 0.8.10
* Pandas Version: 1.4.2
* OpenCV Version: 4.6.0.66

## Setup
To run this project, install it locally using pip
```
#Not Supported Yet.
pip install git+https://github.com/kwmheen357/FaceAnlz.git
```

## Document
### Library Configuration
![library_config](./src/library_config.png)
### FaceAnlz()
> Parent classes of two classes, FaceDetection and FaceMesh, help with basic settings for image recognition.

### FaceMesh()
#### Arguments
> show_process (bool) : Wether show process of detecting the source file. Default False
<br />expansion_rate (float) : Source file expansion rate. Default 1
#### Returns
> eye_traking_list (list) : list of dict. Contains location of eye, fps/frame
### FaceDetection()
#### Arguments
> show_process (bool) : Wether show process of detecting the source file. Default False
<br />expansion_rate (float) : Source file expansion rate. Default 1
#### Returns
> eye_traking_list (list) : list of dict. Contains location of eye, fps/frame

## Example
### FaceMesh()
```
from faceanlz import FaceMesh

file_dir = './data/test.mp4' #Either video or picture is fine.
obj = FaceMesh(file_dir)

eye_list = obj.get_eye_coord() #list of dictionary includes coordinates of eye
```

### FaceDetection()
```
from faceanlz import FaceDetection

file_dir = './data/test.mp4' #Either video or picture is fine.
obj = FaceDetection(file_dir)

eye_list = obj.get_eye_coord() #list of dictionary includes coordinates of eye
```

## Source
This library is inspired by MediaPipe
(https://google.github.io/mediapipe/)