#ifndef GENERATE_DETECTIONS_H
#define GENERATE_DETECTIONS_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <tensorflow/c/c_api.h>
#include <Layer.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include "DetectionNN.h"
#include "TypewithMetadata.h"
#include <stdio.h>
#include <cassert>

using namespace std;
using namespace cv; 


class ImageEncoder{

    public:
        ImageEncoder();
        std::vector<std::vector<DetectionWithFeatureVector>> generateDetections(std::vector<TypewithMetadata<cv::Mat>> *batch_images, tk::dnn::DetectionNN &detNN);
    protected:
    private:
        TensorFlowManager tfm;
        cv::Mat extract_image_patch(const cv::Mat &image, const cv::Rect &bbox, const cv::Size &patch_shape);
        cv::Mat preprocessPatch(const cv::Mat& image);
};

class TensorFlowManager{

    public:
        TensorFlowManager(const std::string& checkpointFilename = "mars-small128.pb");
        ~TensorFlowManager();
        std::vector<float> generateFeatureVector(cv::Mat image);
    protected:
    private:
        std::string checkpointFilename;
        TF_Buffer* tf_buffer;
        TF_Graph* tf_graph;
        TF_Status* status;
        TF_SessionOptions* session_opts;
        TF_Session* session;
        TF_Operation* input_op;
        TF_Output input_var;
        TF_Operation* output_op;
        TF_Output output_var;
        int num_dims_in;
        int num_dims_out;
        int64_t* output_dims;
        int64_t* input_dims;
        bool initializeTensorFlow(const std::string checkpointFilename);
        void deleteSession();
        std::vector<float> runInference(const cv::Mat imagePatch);
        TF_Tensor* createTensorFromMat(const cv::Mat& image);
        TF_Tensor* createOutputTensor();
        void printTensorDims(TF_Tensor* tensor);
};

#endif 