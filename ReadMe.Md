# Project II : Content based Image Retrival 

## Created by Ruohe Zhou and Rucha Pendharkar on 2/8/24

This project is to get familiarized with C++, the OpenCV package, and the mechanics of opening, capturing, manipulating images at the pixel level. It also involved implementing matching, or pattern recognition dealing with textures and Histograms. 

## Introduction 
The overall task for this project is, given a database of images and a target image, find images in the data with similar content. For this project we using both classic features and deep network embeddings. The classic features will be generic characteristics of the images such as color, texture, and their spatial layout. The deep network embedding will be features extracted from a ResNet18 network

## Files

- **Task 1:** Run extractFeatures_program1.cpp followed by the baselineMatching_program2.cpp
- **Task 2:** Run histogramMatching.cpp
- **Task 3:** Run multiHistogram1.cpp followed by multiHistogram2.cpp
- **Task 4:** Run textureColor1.cpp followed by textureColor2.cpp
- **Task 5:** Run featureMatching_usingResNet18.cpp
- **Task 6:** Run featureMatching_usingResNet18.cpp and baselineMatching_program2.cpp for same target images
- **Task 7:** Run extractFeatures_program1.cpp followed customImageRetrival.cpp.
- **Extension:** Run extensionFace.cpp. Make sure the files showFaces.cpp, faceDetect.cpp, and faceDetect_greybg.cpp, kmeans.cpp, kmeans.h, haarcascade_frontalface_alt2.xml are present in the same directory

## Environment 
The scripts were authored using VS Code, and code compilation took place in the Ubuntu 20.04.06 LTS environment, utilizing CMake through the terminal.

## Notes
Please update the file paths according to the structure of your folder. Running cmake once, and directly calling the corresponing executable should run the files
