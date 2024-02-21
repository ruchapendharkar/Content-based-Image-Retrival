/**

Project 2  
  
Created by Ruohe Zhou and Rucha Pendharkar on 2/8/24

This code is used for Task 4. Read features csv generated from textureColor1 file and evaluate 
the similarity between images. 

**/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

// Structure to hold image features
struct ImageFeatures {
    std::string filename;
    std::vector<float> features;
};

// Function to parse features from CSV file
std::vector<ImageFeatures> parseFeatures(const std::string& filename) {
    std::vector<ImageFeatures> featuresList;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open CSV file " << filename << std::endl;
        return featuresList;
    }

    std::string line;
    std::getline(file, line); // Skip header line

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string filename;
        std::getline(iss, filename, ',');

        ImageFeatures imgFeatures;
        imgFeatures.filename = filename;

        std::string feature;
        while (std::getline(iss, feature, ',')) {
            imgFeatures.features.push_back(std::stof(feature));
        }

        featuresList.push_back(imgFeatures);
    }

    return featuresList;
}

// Function to compute similarity score between two feature vectors
float computeSimilarity(const std::vector<float>& features1, const std::vector<float>& features2) {
    float score = 0.0f;
    for (size_t i = 0; i < features1.size(); ++i) {
        score += std::abs(features1[i] - features2[i]);
    }
    return score;
}

int main() {
    // Load features of all images from CSV file
    std::vector<ImageFeatures> allFeatures = parseFeatures("../feature_tc.csv");
    if (allFeatures.empty()) {
        std::cerr << "Error: No features found in CSV file." << std::endl;
        return 1;
    }

    // Select features of image 1
    std::vector<float> featuresOfImage1;
    for (const auto& imgFeatures : allFeatures) {
        if (imgFeatures.filename == "pic.0948.jpg") {
            featuresOfImage1 = imgFeatures.features;
            break;
        }
    }

    if (featuresOfImage1.empty()) {
        std::cerr << "Error: Features of image 1 not found." << std::endl;
        return 1;
    }

    // Compute similarity scores between image 1 and all other images
    std::vector<std::pair<std::string, float>> similarityScores;
    for (const auto& imgFeatures : allFeatures) {
        if (imgFeatures.filename != "pic.0948.jpg") {
            float similarity = computeSimilarity(featuresOfImage1, imgFeatures.features);
            similarityScores.emplace_back(imgFeatures.filename, similarity);
        }
    }

    // Sort images based on similarity scores
    std::sort(similarityScores.begin(), similarityScores.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    // Print top 3 similar images
    std::cout << "Top 3 images similar to image 1:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        std::cout << similarityScores[i].first << " - Similarity Score: " << similarityScores[i].second << std::endl;
    }

    return 0;
}