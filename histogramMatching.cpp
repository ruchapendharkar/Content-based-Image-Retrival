/**

Project 2  
  
Created by Ruohe Zhou and Rucha Pendharkar on 2/8/24

This code is used for Task 2. The code used a whole image RGB histogram using 8 bins 
for each of RGB and histogram intersection as the distance metric.

**/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

void computeChromaticityHistogram(cv::Mat& image, cv::Mat& histogram) {
    // Convert the image to RGB color space
    cv::Mat rgbImage;
    cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);

    // Split the channels
    cv::Mat channels[3];
    cv::split(rgbImage, channels);

    // Assuming red channel for chromaticity
    cv::Mat chromaticityR = channels[0] / (channels[0] + channels[1] + channels[2]);
    chromaticityR *= 255;

    // Ensure the image has a single channel
    if (chromaticityR.channels() != 1) {
        std::cerr << "Error: Images must be single-channel for chromaticity histogram calculation.\n";
        return;
    }

    // Define the number of bins for each channel
    int bins = 8;
    int histSize[] = {bins};

    // Set the range for the channel
    float rRanges[] = {0, 256};
    const float* ranges[] = {rRanges};

    // Compute the histogram
    int channelsForHist[] = {0};
    cv::calcHist(&chromaticityR, 1, channelsForHist, cv::Mat(), histogram, 1, histSize, ranges, true, false);
}


float computeHistogramIntersection(const cv::Mat& hist1, const cv::Mat& hist2) {
    // Compute the histogram intersection
    return cv::compareHist(hist1, hist2, cv::HISTCMP_INTERSECT);
}

std::vector<std::pair<std::string, float>> findMatches(const std::string& targetFilename, const std::string& featureFile, int n) {
    std::vector<std::pair<std::string, float>> distances;

    // Load the target image
    cv::Mat targetImage = cv::imread(targetFilename);
    if (targetImage.empty()) {
        std::cerr << "Error: Unable to read the target image at path " << targetFilename << ".\n";
        return distances;
    }

    // Compute the chromaticity histogram for the target image
    cv::Mat targetHistogram;
    computeChromaticityHistogram(targetImage, targetHistogram);

    // Load features from the feature file
    std::ifstream csvFile(featureFile);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Unable to open the feature file at path " << featureFile << ".\n";
        return distances;
    }

    std::string line;
    std::getline(csvFile, line); // Skip header

    while (std::getline(csvFile, line)) {
        std::istringstream iss(line);
        std::string filename;
        std::getline(iss, filename, ',');

        // Update the path to the images in the ../olympus folder
        std::string imagePath = "../olympus/" + filename;

        if (imagePath == targetFilename) {
            continue; // Skip the target image itself
        }

        // Load the image
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Error: Unable to read the image at path " << imagePath << ".\n";
            continue;
        }

        // Compute the chromaticity histogram for the current image
        cv::Mat currentHistogram;
        computeChromaticityHistogram(image, currentHistogram);

        // Compute the histogram intersection distance
        float distance = computeHistogramIntersection(targetHistogram, currentHistogram);
        distances.push_back({filename, distance});
    }

    // Sort in ascending order based on distances
    std::sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    return {distances.begin(), distances.begin() + n};
}

int main() {
    std::string featureFile = "../features.csv";
    std::string targetFilename = "../olympus/pic.0164.jpg";
    int numMatches = 3;

    auto matches = findMatches(targetFilename, featureFile, numMatches);

    std::cout << "Top " << numMatches << " Matches for " << targetFilename << ":\n";
    for (const auto& match : matches) {
        std::cout << match.first << " - Distance: " << match.second << "\n";
    }

    return 0;
}
