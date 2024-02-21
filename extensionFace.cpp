/**

extensionFace.cpp  
Project 2  
  
Created by Ruohe Zhou and Rucha Pendharkar on 2/8/24

This code is used for the extension. It loops through the directory, and detects faces and then 
performs cartoonization on the detected faces using Kmeans


**/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <filesystem>
#include "faceDetect.cpp"
#include "kmeans.h"
#include "kmeans.cpp"

namespace fs = std::filesystem;

// Function to detect faces in a single image
void detectFacesInImage(cv::Mat &image, std::vector<cv::Rect>& faces) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Detect faces
    detectFaces(gray, faces);
    drawBoxes(image, faces );
}

// Function to perform K-means clustering on a list of images
void performKMeansClustering(const std::vector<cv::Mat>& images, const std::string& outputDirectory) {
    // Set K (number of clusters)
    int K = 7;

    // Iterate through the images and perform K-means clustering
    for (size_t i = 0; i < images.size(); ++i) {
        const cv::Mat& image = images[i];

        // Flatten the image to a vector of cv::Vec3b
        std::vector<cv::Vec3b> data;
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                data.push_back(image.at<cv::Vec3b>(y, x));
            }
        }

        // Allocate space for labels
        int* labels = new int[data.size()];

        // Vector to store cluster means
        std::vector<cv::Vec3b> means;

        // Run the K-means algorithm
        int maxIterations = 10;  // You can adjust this value
        int stopThresh = 0;      // You can adjust this value
        int result = kmeans(data, means, labels, K, maxIterations, stopThresh);

        if (result == 0) {
            // Save the clustered image
            fs::path outputPath = fs::path(outputDirectory) / ("clustered_" + std::to_string(i) + ".jpg");
            cv::Mat clusteredImg(image.size(), image.type());
            for (int y = 0; y < image.rows; ++y) {
                for (int x = 0; x < image.cols; ++x) {
                    clusteredImg.at<cv::Vec3b>(y, x) = means[labels[y * image.cols + x]];
                }
            }

            cv::imwrite(outputPath.string(), clusteredImg);

            std::cout << "K-means clustering completed for Image " << i << std::endl;
        } else {
            std::cerr << "Error: K-means algorithm failed for Image " << i << std::endl;
        }

        // Clean up
        delete[] labels;
    }
}

int main() {
    // Specify the directory containing images
    std::string directoryPath = "/home/rucha/CS5330/Project2/olympus/";
    std::string outputDirectorypath = "/home/rucha/CS5330/Project2/";

    // Vector to store images with faces
    std::vector<cv::Mat> imagesWithFaces;

    // Iterate through the directory and process each image
    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        std::string imagePath = entry.path().string();

        // Load the input image
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Error: Unable to load image: " << imagePath << std::endl;
            continue; // Skip to the next image if loading fails
        }

        // Detect faces in the image
        std::vector<cv::Rect> faces;
        detectFacesInImage(image, faces);

        // Check if faces were detected
        if (!faces.empty()) {
            // Store images with faces
            imagesWithFaces.push_back(image);
        }
    }

    // Perform K-means clustering on the images with faces
    if (!imagesWithFaces.empty()) {
        performKMeansClustering(imagesWithFaces, outputDirectorypath);
    } else {
        std::cout << "No images with faces found in the directory." << std::endl;
    }

    return 0;
}
