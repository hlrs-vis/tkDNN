#include "GenerateDetections.h"
#include "DetectionWithFeatureVector.h"

namespace std {
namespace cv {
namespace tf = tensorflow {

cv::Mat extract_image_patch(const cv::Mat &image, const cv::Rect &bbox, const cv::Size &patch_shape) {
    Yolo::box patched_bbox = bbox;

    if (patch_shape.width > 0 && patch_shape.height > 0) {
        double target_aspect = static_cast<double>(patch_shape.width) / patch_shape.height;
        double new_width = target_aspect * patched_bbox.h;
        patched_bbox.x -= (new_width - patched_bbox.w) / 2;
        patched_bbox.w = new_width;
    }

    // Convert to top left, bottom right
    patched_bbox.w += patched_bbox.x;
    patched_bbox.h += patched_bbox.y;

    // Clip at image boundaries
    patched_bbox.x = std::max(0, patched_bbox.x);
    patched_bbox.y = std::max(0, patched_bbox.y);
    patched_bbox.w = std::min(image.cols - 1, patched_bbox.w);
    patched_bbox.h = std::min(image.rows - 1, patched_bbox.h);

    if (patched_bbox.x >= patched_bbox.w || patched_bbox.y >= patched_bbox.h) {
        return cv::Mat();
    }

    cv::Mat image_patch = image(patched_bbox);
    cv::resize(image_patch, image_patch, patch_shape);
    return image_patch;
}

ImageEncoder::ImageEncoder(const std::string &checkpoint_filename, const std::string &input_name = "images", const std::string &output_name = "features") {
    session = tf.Session()

    
    input_var = tf.get_default_graph().get_tensor_by_name("net/" + input_name + ":0");
    output_var = tf.get_default_graph().get_tensor_by_name("net/" + output_name + ":0");

    assert(session_->Run({}, {input_var}, {}, &input_tensor_).ok());
    assert(session_->Run({}, {output_var}, {}, &output_tensor_).ok());
    feature_dim = output_tensor_.shape().dim_size(1);
    image_shape = cv::Size(input_var.shape().dim_size(2), input_var.shape().dim_size(1));
}

ImageEncoder::call(const cv::Mat &data) {
    size_t data_lenght = data.size();
    vector<vector<float>> out(data_lenght , std::vector<float>(feature_dim, 0.0f));
    
}




cv::Mat createBoxEncoder(const std::string &model_filename, const std::string &input_name = "images", const std::string &output_name = "features") {
    ImageEncoder image_encoder;
    image_encoder.ImageEncoder(model_filename, input_name, output_name);
    cv::Size image_shape = image_encoder.image_shape;

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

        return image_encoder.call(image_patches);
    };

    return encoder;
}

std::vector<DetectionWithFeatureVector> generateDetections(std:function encoder, cv:Mat *batch_images, tk::dnn::detectionNN &detNN) { 
    
    std::vector<DetectionWithFeatureVector> detections_out;
    for (int bi = 0; bi < detNN.batchDetected.size(); ++bi){        // Iterate the frames in one batch
        vector<cv::Rect> boxes;
        for (int i = 0; i < detNN.batchDetected[bi].size(); i++){   // Iterate the detections in one frame
            tk::dnn::box b;
            float gX = -1, gY = -1, gZ = -1;
            b = detNN.batchDetected[bi][i];                
            boxes.push_back(cv::Rect(static_cast<int>(b.x), static_cast<int>(b.y), static_cast<int>(b.w), static_cast<int>(b.h))); // Safe the x,y-coordinates, width and height
            detections_out.push_back({(*batch_images)[bi].frame_id, b.cl, b.x, b.y, b.w, b.h, b.prob, gX, gV, gZ, 0}); // Safe the detections in the MOT challenge format
        }
        vector<vector<float>> features = encoder(batch_images[bi].data, std::vector boxes); // Call encoder with all detections for one frame
        
        // Fill the feature vector
        for (int i = 0; i < features.size(); i++) {     
            detections_out.feature_vector = features[i];
        }
        boxes.clear();
        detections_in.clear();
    }
    return detections_out;
}

} } }

