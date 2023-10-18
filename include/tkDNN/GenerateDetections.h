#ifndef GENERATE_DETECTIONS_H
#define GENERATE_DETECTIONS_H

cv::Mat extract_image_patch(const cv::Mat &image, const cv::Rect &bbox, const cv::Size &patch_shape)

class ImageEncoder{

    public:
        ImageEncoder();
        void initiate();
        
    protected:

    private:
        

};

class Detector{

    public:
        auto encoder = create_box_encoder(model_filename, "images", "features", 32);
        generate_detections(encoder, mot_dir, output_dir, detection_dir);

    protected:

    private:
        const std::string model_filename;

};