#ifndef DETECTIONWITHFEATUREVECTOR_H
#define DETECTIONWITHFEATUREVECTOR_H

#include <stdlib.h>
#include <vector>

struct DetectionWithFeatureVector {
    long long int frame_id;
    int bbox_x;
    int bbox_y;
    int bbox_w;
    int bbox_h;
    float probability;
    std::vector<float> feature_vector;

};


#endif