/**

customImageRetrival.cpp  
Project 2  
  
Created by Ruohe Zhou and Rucha Pendharkar on 2/8/24

This code is used for Task 7. The code aims to provide a combined similarity
measure for images with red stuffed animals by considering both texture and color features. 


**/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <numeric>


float computeCosineDistance(const std::vector<float>& feature1, const std::vector<float>& feature2) {
    float dotProduct = 0.0;
    float normFeature1 = 0.0;
    float normFeature2 = 0.0;
    float cosdistance = 0.0;

    for (size_t i = 0; i < feature1.size(); ++i) {
        dotProduct += feature1[i] * feature2[i];
        normFeature1 += std::pow(feature1[i], 2);
        normFeature2 += std::pow(feature2[i], 2);
    }

    normFeature1 = std::sqrt(normFeature1);
    normFeature2 = std::sqrt(normFeature2);

    // Avoid division by zero
    if (normFeature1 == 0.0 || normFeature2 == 0.0) {
        std::cerr << "Error: Division by zero in cosine distance calculation.\n";
        return -1.0;  // Invalid distance
    }
    cosdistance = dotProduct / (normFeature1 * normFeature2);
    return std::acos(cosdistance);
}

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

//Find Texture matches using ResNet features
std::vector<std::pair<std::string, float>> findTextureMatches(const std::string& targetFilename, const std::string& featureFile, int n) {
    std::vector<std::pair<std::string, float>> distances;

    // Load features from the feature file
    std::ifstream csvFile(featureFile);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Unable to open the feature file at path " << featureFile << ".\n";
        return distances;
    }

    std::string line;
    std::getline(csvFile, line); // Skip header

    // Find target features directly from the CSV file
    std::vector<float> targetFeatures;
    while (std::getline(csvFile, line)) {
        std::istringstream iss(line);
        std::string filename;
        std::getline(iss, filename, ',');

        if (filename == targetFilename) {
            float featureValue;
            while (iss >> featureValue) {
                targetFeatures.push_back(featureValue);
                iss.ignore(); // Ignore comma
            }
            break; // Target image found, break the loop
        }
    }

    // Verify if the target image was found in the CSV file
    if (targetFeatures.empty()) {
        std::cerr << "Error: Target image not found in the feature file.\n";
        return distances;
    }

    csvFile.clear();
    csvFile.seekg(0, std::ios::beg);

    // Process the rest of the CSV file to calculate distances
    std::getline(csvFile, line); // Skip header

    while (std::getline(csvFile, line)) {
        std::istringstream iss(line);
        std::string filename;
        std::getline(iss, filename, ',');

        if (filename == targetFilename) {
            continue; // Skip the target image itself
        }

        std::vector<float> features;
        float featureValue;
        while (iss >> featureValue) {
            features.push_back(featureValue);
            iss.ignore(); // Ignore comma
        }

        float distance = computeCosineDistance(targetFeatures, features);
        distances.push_back({filename, distance});
    }
   return distances;
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
        std::string imagePath = "/home/rucha/CS5330/Project2/olympus/" + filename;

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
    return distances;
}

std::vector<std::pair<std::string, float>> findCombinedMatches(const std::string& targetFilename, const std::string& targetTextureFilename, const std::string& textureFile, const std::string& colorHistFile, int n, float textureWeight, float colorWeight) {
    std::vector<std::pair<std::string, float>> combinedDistances;
    auto TextMatches = findTextureMatches(targetTextureFilename, textureFile, n);
    auto ColorMatches = findMatches(targetFilename, colorHistFile, n);

    // Normalize distances to the range [0, 1]
    float maxTextureDistance = TextMatches.front().second;
    float maxColorDistance = ColorMatches.front().second;

    for (auto& TextMatch : TextMatches) {
        TextMatch.second /= maxTextureDistance;
    }

    for (auto& ColorMatch : ColorMatches) {
        ColorMatch.second /= maxColorDistance;
    }
    size_t loopLimit = std::min({TextMatches.size(), ColorMatches.size()});

    // Combine the normalized distances using weighted sum
    for (int i = 0; i < loopLimit; ++i) {
        float combinedDistance = textureWeight * TextMatches[i].second + colorWeight * ColorMatches[i].second;
        combinedDistance = 1/combinedDistance;
        combinedDistances.push_back({TextMatches[i].first, combinedDistance});
    }

    // Sort the combined distances
    std::sort(combinedDistances.begin(), combinedDistances.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    return combinedDistances;
}

int main() {
    std::string textureFile = "/home/rucha/CS5330/Project2/ResNet18_olym.csv";
    std::string targetTextureFilename = "pic.0930.jpg";

    std::string colorHistFile = "/home/rucha/CS5330/Project2/feature.csv";
    std::string targetFilename = "/home/rucha/CS5330/Project2/olympus/pic.0930.jpg";
    int numMatches = 5;

    float textureWeight = 0.7; // Weight for texture matching
    float colorWeight = 0.3;   // Weight for color matching

    auto combinedMatches = findCombinedMatches(targetFilename, targetTextureFilename, textureFile, colorHistFile, numMatches, textureWeight, colorWeight);

    std::cout << "Top " << numMatches << " Combined Matches for " << targetTextureFilename << ":\n";

    // Output only the top 5 matches
    for (int i = 0; i < numMatches && i < combinedMatches.size(); ++i) {
        const auto& combinedMatch = combinedMatches[i];
        std::cout << "Filename: " << combinedMatch.first << ",  Distance: " << combinedMatch.second << "\n";
    }

    return 0;
}


