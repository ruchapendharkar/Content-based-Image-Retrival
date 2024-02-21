/**

Project 2  
  
Created by Ruohe Zhou and Rucha Pendharkar on 2/8/24

This code is used for Task 4. The code calculate the Sobel magnitude image and 
use a histogram of gradient magnitudes as texture feature
and compute a csv file with all features as well as file name.

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
    std::vector<float> colorHistogram;
    std::vector<float> textureHistogram;
};


// Compute whole image color histogram
void computeColorHistogram(const cv::Mat& image, std::vector<float>& histogram) {
    // Convert image to HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    // Split the channels
    std::vector<cv::Mat> channels;
    cv::split(hsvImage, channels);

    // Compute histogram
    int histSize[] = {256};
    float range[] = {0, 256};
    const float* histRange[] = {range};
    int channelsIdx[] = {0, 1, 2};

    for (int i = 0; i < 3; ++i) {
        cv::Mat channel8U;
        channels[i].convertTo(channel8U, CV_8U);

        // Calculate histogram for each channel and accumulate values
        cv::Mat hist;
        cv::calcHist(&channel8U, 1, nullptr, cv::Mat(), hist, 1, histSize, histRange);
        histogram.insert(histogram.end(), hist.begin<float>(), hist.end<float>());
    }

    // Normalize histogram
    cv::normalize(histogram, histogram, 0, 1, cv::NORM_MINMAX);
}





// Compute texture histogram using Sobel operator
void computeTextureHistogram(const cv::Mat& image, std::vector<float>& histogram) {
    // Convert image to grayscale
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // Compute gradients using Sobel operator
    cv::Mat gradX, gradY;
    cv::Sobel(grayImage, gradX, CV_32F, 1, 0);
    cv::Sobel(grayImage, gradY, CV_32F, 0, 1);

    // Compute gradient magnitude
    cv::Mat magImage;
    cv::magnitude(gradX, gradY, magImage);

    // Convert gradient magnitude to CV_8U for histogram calculation
    cv::Mat magImage8U;
    magImage.convertTo(magImage8U, CV_8U);

    // Compute histogram of gradient magnitudes
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange[] = {range};
    cv::calcHist(&magImage8U, 1, 0, cv::Mat(), histogram, 1, &histSize, histRange);
    cv::normalize(histogram, histogram, 0, 1, cv::NORM_MINMAX);
}

// Extract features from images in a directory and save them to a CSV file
void extractFeaturesAndSave(const std::string& inputDir, const std::string& outputFile) {
    std::vector<ImageFeatures> featuresList;

    // Iterate over images in the input directory
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            // Read image
            cv::Mat image = cv::imread(entry.path().string());
            if (image.empty()) {
                std::cerr << "Error: Unable to read image at path " << entry.path() << std::endl;
                continue;
            }

            // Ensure the image has 3 channels (BGR)
            if (image.channels() != 3) {
                std::cerr << "Error: Image must have 3 channels (BGR)." << std::endl;
                continue;
            }

            // Compute color histogram
            std::vector<float> colorHistogram;
            computeColorHistogram(image, colorHistogram);

            // Ensure the color histogram is not empty
            if (colorHistogram.empty()) {
                std::cerr << "Error: Color histogram is empty for image " << entry.path() << std::endl;
                continue;
            }

            // Compute texture histogram
            std::vector<float> textureHistogram;
            computeTextureHistogram(image, textureHistogram);

            // Ensure the texture histogram is not empty
            if (textureHistogram.empty()) {
                std::cerr << "Error: Texture histogram is empty for image " << entry.path() << std::endl;
                continue;
            }

            // Extract filename from path
            std::string filename = entry.path().filename().string();

            // Create ImageFeatures struct and add to list
            ImageFeatures imgFeatures;
            imgFeatures.filename = filename;
            imgFeatures.colorHistogram = colorHistogram;
            imgFeatures.textureHistogram = textureHistogram;
            featuresList.push_back(imgFeatures);
        }
    }

    // Write features to CSV file
    std::ofstream csvFile(outputFile);
    csvFile << "filename,";
    for (size_t i = 0; i < featuresList[0].colorHistogram.size(); ++i) {
        csvFile << "color_feature_" << i << ",";
    }
    for (size_t i = 0; i < featuresList[0].textureHistogram.size(); ++i) {
        csvFile << "texture_feature_" << i << ",";
    }
    csvFile << "\n";

    for (const auto& row : featuresList) {
        csvFile << row.filename << ",";
        for (const auto& feature : row.colorHistogram) {
            csvFile << feature << ",";
        }
        for (const auto& feature : row.textureHistogram) {
            csvFile << feature << ",";
        }
        csvFile << "\n";
    }
}


int main() {
    // Input directory containing images
    std::string inputDirectory = "../olympus";

    // Output CSV file to save features
    std::string outputFeatureFile = "../feature_tc.csv";

    // Extract features from images and save to CSV file
    extractFeaturesAndSave(inputDirectory, outputFeatureFile);

    return 0;
}
