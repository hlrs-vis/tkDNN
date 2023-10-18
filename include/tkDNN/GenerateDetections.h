#ifndef GENERATE_DETECTIONS_H
#define GENERATE_DETECTIONS_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>

namespace tens
cv::Mat extract_image_patch(const cv::Mat &image, const cv::Rect &bbox, const cv::Size &patch_shape);

class ImageEncoder{

    public:
        ImageEncoder();
        void encode();
        
    protected:

    private:
        

};

cv::Mat create_box_encoder(const std::string &model_filename, const std::string &input_name = "images", const std::string &output_name = "features", const int batch_size = 32);
