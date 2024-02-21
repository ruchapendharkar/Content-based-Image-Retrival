/**

featureMatching_usingResNet18.cpp  
Project 2  
  
Created by Ruohe Zhou and Rucha Pendharkar on 2/8/24

This code is used for Task 5. The code aims return top three similar images computed from the
features with DNN Embeddings. 


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

std::vector<std::pair<std::string, float>> findMatches(const std::string& targetFilename, const std::string& featureFile, int n) {
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

    std::sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) {
        return a.second < b.second; // Sort in ascending order of distance
    });

    return {distances.begin(), distances.begin() + n};
}

int main() {
    std::string featureFile = "/home/rucha/CS5330/Project2/ResNet18_olym.csv"; 
    std::string targetFilename = "pic.0734.jpg";
    int numMatches = 3;

    auto matches = findMatches(targetFilename, featureFile, numMatches);

    std::cout << "Top " << numMatches << " Matches for " << targetFilename << ":\n";
    for (const auto& match : matches) {
        std::cout << "Filename: " << match.first << ",  Distance: " << match.second << "\n";
    }

    return 0;
}
