#ifndef DETECTIONWITHFEATUREVECTOR_H
#define DETECTIONWITHFEATUREVECTOR_H

#include <stdlib.h>
#include <vector.h>

struct DetectionWithFeatureVector {
    long long int frame_id;
    std::int detections_class;
    std::float bbox_x;
    std::float bbox_y;
    std::float bbox_w;
    std::float bbox_h;
    std::float probability; 
    std::float global_x;
    std::float global_y;
    std::float gloabel_z;
    vector<std::float> feature_vector;

};


#endif