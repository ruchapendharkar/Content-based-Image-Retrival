/**

Project 2  
  
Created by Ruohe Zhou and Rucha Pendharkar on 2/8/24

This code is used for Task 3. The code used two RGB histograms, 
representing the top and bottom halves of the image, 
using 8 bins for each of RGB and histogram intersection as the distance metric.


**/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

// Structure to hold image features
struct ImageFeatures {
    std::string filename;
    std::vector<float> featuresTop;
    std::vector<float> featuresBottom;
};

// Compute RGB histograms for top and bottom halves of an image
void computeFeatures(cv::Mat& image, std::vector<float>& featuresTop, std::vector<float>& featuresBottom) {
    // Define regions of interest for top and bottom halves
    cv::Rect topRect(0, 0, image.cols, image.rows / 2);
    cv::Rect bottomRect(0, image.rows / 2, image.cols, image.rows / 2);

    // Split the image into regions
    cv::Mat topRegion = image(topRect);
    cv::Mat bottomRegion = image(bottomRect);

    // Compute histograms for each region
    int bins = 8;
    int histSize[] = {bins, bins, bins};
    float range[] = {0, 256}; // Range for each channel
    const float* ranges[] = {range, range, range};
    int channels[] = {0, 1, 2}; // All channels

    // Compute histogram for top region
    cv::Mat histTop;
    cv::calcHist(&topRegion, 1, channels, cv::Mat(), histTop, 3, histSize, ranges);
    cv::normalize(histTop, histTop, 0, 1, cv::NORM_MINMAX);
    histTop = histTop.reshape(1, 1); // Flatten histogram

    // Compute histogram for bottom region
    cv::Mat histBottom;
    cv::calcHist(&bottomRegion, 1, channels, cv::Mat(), histBottom, 3, histSize, ranges);
    cv::normalize(histBottom, histBottom, 0, 1, cv::NORM_MINMAX);
    histBottom = histBottom.reshape(1, 1); // Flatten histogram

    // Convert histograms to vector<float>
    featuresTop.assign(histTop.begin<float>(), histTop.end<float>());
    featuresBottom.assign(histBottom.begin<float>(), histBottom.end<float>());
}

// Compute histogram intersection distance between two histograms
float computeHistogramIntersection(const std::vector<float>& hist1, const std::vector<float>& hist2) {
    float distance = 0.0f;
    for (size_t i = 0; i < hist1.size(); ++i) {
        distance += std::min(hist1[i], hist2[i]);
    }
    return distance;
}

// Extract features from images in a directory and save them to a CSV file
void extractFeaturesAndSave(const std::string& inputDir, const std::string& outputFile) {
    std::vector<ImageFeatures> featuresList;

    // Iterate over images in the input directory
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            // Read image
            cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
            if (image.empty()) {
                std::cerr << "Error: Unable to read image at path " << entry.path() << std::endl;
                continue;
            }

            // Compute features
            std::vector<float> featuresTop, featuresBottom;
            computeFeatures(image, featuresTop, featuresBottom);

            // Extract filename from path
            std::string filename = entry.path().filename().string();

            // Create ImageFeatures struct and add to list
            ImageFeatures imgFeatures;
            imgFeatures.filename = filename;
            imgFeatures.featuresTop = featuresTop;
            imgFeatures.featuresBottom = featuresBottom;
            featuresList.push_back(imgFeatures);
        }
    }

    // Write features to CSV file
    std::ofstream csvFile(outputFile);
    csvFile << "filename,";
    for (size_t i = 0; i < featuresList[0].featuresTop.size(); ++i) {
        csvFile << "top_feature_" << i << ",";
    }
    for (size_t i = 0; i < featuresList[0].featuresBottom.size(); ++i) {
        csvFile << "bottom_feature_" << i << ",";
    }
    csvFile << "\n";

    for (const auto& row : featuresList) {
        csvFile << row.filename << ",";
        for (const auto& feature : row.featuresTop) {
            csvFile << feature << ",";
        }
        for (const auto& feature : row.featuresBottom) {
            csvFile << feature << ",";
        }
        csvFile << "\n";
    }
}

int main() {
    // Input directory containing images
    std::string inputDirectory = "../olympus";

    // Output CSV file to save features
    std::string outputFeatureFile = "../feature_multi.csv";

    // Extract features from images and save to CSV file
    extractFeaturesAndSave(inputDirectory, outputFeatureFile);

    return 0;
}
