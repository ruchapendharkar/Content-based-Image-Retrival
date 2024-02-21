#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "faceDetect.h"

int detectFaces_greybg(cv::Mat &grey, std::vector<cv::Rect> &faces) {
    // a static variable to hold a half-size image
    static cv::Mat half;

    // a static variable to hold the classifier
    static cv::CascadeClassifier face_cascade;

    // the path to the haar cascade file
    static cv::String face_cascade_file(FACE_CASCADE_FILE);

    if (face_cascade.empty()) {
        if (!face_cascade.load(face_cascade_file)) {
            printf("Unable to load face cascade file\n");
            printf("Terminating\n");
            exit(-1);
        }
    }

    // clear the vector of faces
    faces.clear();

    // cut the image size in half to reduce processing time
    cv::resize(grey, half, cv::Size(grey.cols / 2, grey.rows / 2));

    if (half.channels() == 1) {
        cv::equalizeHist(half, half);
    } else {
    // Convert the multi-channel image to grayscale before equalization
        cv::Mat gray;
        cv::cvtColor(half, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, half);
    }

    // apply the Haar cascade detector
    face_cascade.detectMultiScale(half, faces);

    // adjust the rectangle sizes back to the full-size image
    for (int i = 0; i < faces.size(); i++) {
        faces[i].x *= 2;
        faces[i].y *= 2;
        faces[i].width *= 2;
        faces[i].height *= 2;
    }

    return 0;
}

/* Draws rectangles into frame given a vector of rectangles
   
   Arguments:
   cv::Mat &frame - image in which to draw the rectangles
   std::vector<cv::Rect> &faces - standard vector of cv::Rect rectangles
   int minSize - ignore rectangles with a width smaller than this argument
   float scale - scale the rectangle values by this factor (in case the frame is different than the source image)
 */
int drawBoxes_greybg(cv::Mat &frame, std::vector<cv::Rect> &faces, int minWidth, float scale) {
    // The color to draw, you can change it here (B, G, R)
    cv::Scalar wcolor(170, 120, 110);

    for (int i = 0; i < faces.size(); i++) {
        if (faces[i].width > minWidth) {
            cv::Rect face(faces[i]);
            face.x *= scale;
            face.y *= scale;
            face.width *= scale;
            face.height *= scale;

            // Create a mask for the face region
            cv::Mat mask(frame.size(), CV_8UC1, cv::Scalar(0));
            cv::rectangle(mask, face, cv::Scalar(255), cv::FILLED);

            // Convert the mask to a 3-channel image
            cv::Mat maskBGR;
            cv::cvtColor(mask, maskBGR, cv::COLOR_GRAY2BGR);

            // Copy the original frame and the grey frame
            cv::Mat originalCopy = frame.clone();
            cv::Mat greyFrame;
            cv::cvtColor(originalCopy, greyFrame, cv::COLOR_BGR2GRAY);

            // Apply the mask to the original frame
            originalCopy.copyTo(frame, maskBGR);

            // Convert the grey frame to a 3-channel image
            cv::Mat greyBGR;
            cv::cvtColor(greyFrame, greyBGR, cv::COLOR_GRAY2BGR);

            // Apply the inverted mask to the grey frame
            greyBGR.copyTo(frame, ~maskBGR);

            // Draw the rectangle on the original frame
            cv::rectangle(frame, face, wcolor, 3);
        }
    }

    return 0;
}
