#ifndef DETECTIONWITHFEATUREVECTOR_H
#define DETECTIONWITHFEATUREVECTOR_H

#include <stdlib.h>
#include <vector>

struct DetectionWithFeatureVector {
    long long int frame_id;
    int detection_class;
    float bbox_x;
    float bbox_y;
    float bbox_w;
    float bbox_h;
    float probability; 
    float global_x;
    float global_y;
    float global_z;
    std::vector<float> feature_vector;

};


#endif