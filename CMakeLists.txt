cmake_minimum_required(VERSION 3.0.0)
project(project1 VERSION 0.1.0)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Boost and its components
find_package(Boost REQUIRED COMPONENTS filesystem)

# Include directories for OpenCV and Boost
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# Add the executable and link against OpenCV and Boost libraries
add_executable(extractFeatures_program1 extractFeatures_program1.cpp)
target_link_libraries(extractFeatures_program1 ${OpenCV_LIBS} ${Boost_LIBRARIES})
add_executable(baselineMatching_program2 baselineMatching_program2.cpp)
target_link_libraries(baselineMatching_program2 ${OpenCV_LIBS} ${Boost_LIBRARIES})
add_executable(histogramMatching histogramMatching.cpp)
target_link_libraries(histogramMatching ${OpenCV_LIBS} ${Boost_LIBRARIES})
add_executable(multiHistogram1 multiHistogram1.cpp)
target_link_libraries(multiHistogram1 ${OpenCV_LIBS} ${Boost_LIBRARIES})
add_executable(multiHistogram2 multiHistogram2.cpp)
target_link_libraries(multiHistogram2 ${OpenCV_LIBS} ${Boost_LIBRARIES})
add_executable(textureColor1 textureColor1.cpp)
target_link_libraries(textureColor1 ${OpenCV_LIBS} ${Boost_LIBRARIES})
add_executable(textureColor2 textureColor2.cpp)
target_link_libraries(textureColor2 ${OpenCV_LIBS} ${Boost_LIBRARIES})
add_executable(extensionFace extensionFace.cpp)
target_link_libraries(extensionFace ${OpenCV_LIBS} ${Boost_LIBRARIES})
add_executable(customImageRetrival customImageRetrival.cpp)
target_link_libraries(customImageRetrival ${OpenCV_LIBS} ${Boost_LIBRARIES})
add_executable(featureMatching_usingResNet18 featureMatching_usingResNet18.cpp)
target_link_libraries(featureMatching_usingResNet18 ${OpenCV_LIBS} ${Boost_LIBRARIES})
