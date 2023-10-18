#include "GenerateDetections.h"

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

ImageEncoder::ImageEncoder(){

}

void ImageEncoder::initiate(){


}

Detector::Detector(){}

std::function<std::vector<float>(const cv::Mat &, const std::vector<cv::Rect> &)> Detector::create_box_encoder(const std::string &model_filename, const std::string &input_name = "images", const std::string &output_name = "features", int batch_size = 32)