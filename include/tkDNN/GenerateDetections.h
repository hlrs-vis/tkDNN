#ifndef GENERATE_DETECTIONS_H
#define GENERATE_DETECTIONS_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <layer.h>
#include <nlohmann/json.hpp>

namespace tf = tensorflow {


cv::Mat extract_image_patch(const cv::Mat &image, const Yolo::box &bbox, const cv::Size &patch_shape);

class ImageEncoder{

    public:
        session;
        tf::Tensor input_var;
        tf::Tensor output_var;
        feature_dim;
        image_shape;
        ImageEncoder(const std::string &checkpoint_filename, const std::string &input_name, const std::string &output_name);
        call(const cv::Mat &data, int batch_size=32);
        
    protected:

    private:
        
};

// create_box_encoder creates the function to encode detections
cv::Mat create_box_encoder(const std::string &model_filename, const std::string &input_name, const std::string &output_name , const int batch_size);

cv::Mat generateDetections(std:function encoder, cv:Mat *batch_images, tk::dnn::detectionNN &detNN);

}

