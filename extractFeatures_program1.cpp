/**

Project 2  
  
Created by Ruohe Zhou and Rucha Pendharkar on 2/8/24

This code is used for Task 1. The code extract features and form a features.csv file


**/
#include <cmath>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
namespace fs = std::filesystem;


 void computeFeatures(cv::Mat& image, std::vector<float>& features) {
    
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    // Compute the center region
    int regionSize = 7;
    int startX = image.cols / 2 - regionSize / 2;
    int startY = image.rows / 2 - regionSize / 2;

    // Clamp start coordinates to ensure region is within the image bounds
    startX = std::max(0, startX);
    startY = std::max(0, startY);
    int endX = std::min(image.cols, startX + regionSize);
    int endY = std::min(image.rows, startY + regionSize);

    // Crop the center region
    cv::Mat centerRegion = image(cv::Rect(startX, startY, endX - startX, endY - startY));

    // Detect keypoints
    std::vector<cv::KeyPoint> keypoints;
    orb->detect(centerRegion, keypoints);

    cv::Mat descriptors;
    orb->compute(centerRegion, keypoints, descriptors);

    // Convert descriptors to float vector
    features.clear();
    features.reserve(descriptors.rows * descriptors.cols);
    for (int i = 0; i < descriptors.rows; ++i) {
        for (int j = 0; j < descriptors.cols; ++j) {
            features.push_back(static_cast<float>(descriptors.at<uchar>(i, j)));
        }
    }
}


void extractFeaturesAndSave(const std::string& inputDir, const std::string& outputFile) {
    std::vector<std::pair<std::string, std::vector<float>>> featuresList;

    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
            std::vector<float> features;
            computeFeatures(image, features);

            // Extract filename from the path
            std::string filename = entry.path().filename().string();

            featuresList.push_back({filename, features});
        }
    }

    std::ofstream csvFile(outputFile);
    csvFile << "filename,";
    for (int i = 0; i < featuresList[0].second.size(); ++i) {
        csvFile << "feature_" << i << ",";
    }
    csvFile << "\n";

    for (const auto& row : featuresList) {
        csvFile << row.first << ",";
        for (const auto& feature : row.second) {
            csvFile << feature << ",";
        }
        csvFile << "\n";
    }
}

int main() {
    std::string inputDirectory = "../olympus";
    std::string outputFeatureFile = "../features.csv";

    extractFeaturesAndSave(inputDirectory, outputFeatureFile);

    return 0;
}

