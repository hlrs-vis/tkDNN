#ifndef JSON_COMPOSER_H
#define JSON_COMPOSER_H

#include "TypewithMetadata.h"
#include "DetectionNN.h"

using namespace cv;
using namespace tk; 


// JSON format:
//{
// "frame_id":8990, "frame_time":1631555332.334568
// "objects":[
//  {"class_id":4, "name":"aeroplane", "relative coordinates":{"center_x":0.398831, "center_y":0.630203, "width":0.057455, "height":0.020396}, "confidence":0.793070},
//  {"class_id":14, "name":"bird", "relative coordinates":{"center_x":0.398831, "center_y":0.630203, "width":0.057455, "height":0.020396}, "confidence":0.265497}
// ]
//},


class JsonComposer
{

public:
    JsonComposer();
    char* detection_to_json(std::vector<TypewithMetadata<cv::Mat>> *batch_images, tk::dnn::DetectionNN &detNN, char *filename);
    void setResolution(int width, int height);

protected:

private:
    int m_width = 0;
    int m_height = 0;

};

#endif /*JSON_COMPOSER_H*/