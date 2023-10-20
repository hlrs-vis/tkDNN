#include "GenerateDetections.h"

namespace std {
namespace cv {
namespace tf = tensorflow {

cv::Mat extract_image_patch(const cv::Mat &image, const cv::Rect &bbox, const cv::Size &patch_shape) {
    cv::Rect patched_bbox = bbox;

    if (patch_shape.width > 0 && patch_shape.height > 0) {
        double target_aspect = static_cast<double>(patch_shape.width) / patch_shape.height;
        double new_width = target_aspect * patched_bbox.height;
        patched_bbox.x -= (new_width - patched_bbox.width) / 2;
        patched_bbox.width = new_width;
    }

    // Convert to top left, bottom right
    patched_bbox.width += patched_bbox.x;
    patched_bbox.height += patched_bbox.y;

    // Clip at image boundaries
    patched_bbox.x = std::max(0, patched_bbox.x);
    patched_bbox.y = std::max(0, patched_bbox.y);
    patched_bbox.width = std::min(image.cols - 1, patched_bbox.width);
    patched_bbox.height = std::min(image.rows - 1, patched_bbox.height);

    if (patched_bbox.x >= patched_bbox.width || patched_bbox.y >= patched_bbox.height) {
        return cv::Mat();
    }

    cv::Mat image_patch = image(patched_bbox);
    cv::resize(image_patch, image_patch, patch_shape);
    return image_patch;
}

ImageEncoder::ImageEncoder(const std::string &checkpoint_filename, const std::string &input_name = "images", const std::string &output_name = "features") {
    session = tf.Session()
    status = tf

    input_var_ = input_name + ":0";
    output_var_ = output_name + ":0";

    assert(session_->Run({}, {input_var_}, {}, &input_tensor_).ok());
    assert(session_->Run({}, {output_var_}, {}, &output_tensor_).ok());

    feature_dim = output_tensor_.shape().dim_size(1);
    image_shape = cv::Size(input_tensor_.shape().dim_size(2), input_tensor_.shape().dim_size(1));
}




cv::Mat createBoxEncoder(const std::string &model_filename, const std::string &input_name = "images", const std::string &output_name = "features", const int batch_size = 32) {
    ImageEncoder image_encoder(model_filename, input_name, output_name);
    cv::Size image_shape = image_encoder.getImageShape();

    auto encoder = [&](const cv::Mat& image, const std::vector<cv::Rect>& boxes) {
        std::vector<cv::Mat> image_patches;
        for (const cv::Rect& box : boxes) {
            cv::Mat patch = extract_image_patch(image, box, cv::Size(image_shape.width, image_shape.height));
            if (patch.empty()) {
                std::cout << "WARNING: Failed to extract image patch: " << box << "." << std::endl;
                patch = cv::Mat(image_shape, CV_8UC3); 
                cv::randu(patch, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
            }
            image_patches.push_back(patch);
        }

        return image_encoder.encodeImagePatches(image_patches, batch_size);
    };

    return encoder;
}

cv::Mat generateDetections(std:function encoder, cv:Mat *batch_images, tk::dnn::detectionNN &detNN) {
    
    for (int bi = 0; bi < detNN.batchDetected.size(); ++bi){        //iterate the frames in one batch
        for (int i = 0; i < detNN.batchDetected[bi].size(); i++){   //iterate the detections in one frame
            b = detNN.batchDetected[bi][i];                         //create 
            vector<cv::Rect> boxes;
            boxes.push_back(cv::Rect(static_cast<int>(b.x), static_cast<int>(b.y), static_cast<int>(b.w), static_cast<int>(b.h))); //safe the 
            
        }
        vector<cv::float> features = encoder(batch_image[bi].data, std::vector boxes); //call encoder with all detections for one frame
    }
    

    detections 
    
}

} } }

