#ifndef GENERATE_DETECTIONS_H
#define GENERATE_DETECTIONS_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <tensorflow/c/c_api.h>
#include <Layer.h>
#include <nlohmann/json.hpp>
#include "DetectionNN.h"
#include "TypewithMetadata.h"

using namespace std;
using namespace cv; 

cv::Mat extract_image_patch(const cv::Mat &image, const cv::Rect &bbox, const cv::Size &patch_shape);

class ImageEncoder{

    public:
        ImageEncoder();
        int initialization(const std::string &checkpoint_filename, const std::string &input_name, const std::string &output_name);
        void endingSession();
        vector<vector<float>> call(const cv::Mat &data, int batch_size=32);
        
    protected:

    private:
        
};

// create_box_encoder creates the function to encode detections
cv::Mat create_box_encoder(const std::string &model_filename, const std::string &input_name, const std::string &output_name , const int batch_size);

cv::Mat generateDetections(std::function<auto(cv::Mat, vector<cv::Rect>)> encoder, std::vector<TypewithMetadata<cv::Mat>> *batch_images, tk::dnn::detectionNN &detNN);

#endif 