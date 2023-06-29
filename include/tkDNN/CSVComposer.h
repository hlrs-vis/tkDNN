#ifndef CSV_COMPOSER_H
#define CSV_COMPOSER_H

#include "TypewithMetadata.h"
#include "DetectionNN.h"
#include <iostream>
#include <stdlib.h>
#include <fstream>

// CSV format:
// <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

class CSVComposer{

    public:
        CSVComposer();
        void initiate(const std::string &csvFileName, std::ofstream &csvFileStream, const std::string &inputVideo);
        void detectionToCsv(std::vector<TypewithMetadata<cv::Mat>> *batch_images, tk::dnn::DetectionNN &detNN, std::ofstream &csvFileStream);
        void setResolution(int width, int height);
    protected:

    private:
        std::string inputVideoName;
        std::string myFileName;
        int width = 0;
        int height = 0;

};

#endif /*CSV_COMPOSER_H*/