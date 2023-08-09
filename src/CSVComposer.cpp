#include "CSVComposer.h"
#include <iostream>

CSVComposer::CSVComposer(){

}

void CSVComposer::setResolution(int width, int height){
    width = width;
    height = height;
}

void CSVComposer::detectionToCsv(std::vector<TypewithMetadata<cv::Mat>> *batch_images, tk::dnn::DetectionNN &detNN, std::ofstream &csvFileStream){
    float bX, bY, bW, bH, gX = -1, gY = -1, gZ = -1;
    std::string detClass;
    tk::dnn::box b;
    for (int bi = 0; bi < detNN.batchDetected.size(); ++bi){
        for (int i = 0; i < detNN.batchDetected[bi].size(); i++){
            
            b = detNN.batchDetected[bi][i];
            detClass = detNN.classesNames[b.cl];
            bX = b.x;
            bY = b.y;
            bW = b.w;
            bH = b.h;
            csvFileStream << (*batch_images)[bi].frame_id << ", " << b.cl << ", " << bX << ", " << bY << ", " << bW << ", " << bH << ", " << b.prob << ", " << gX << ", " << gY << ", " << gZ << "\n";
        }

    }
}

void CSVComposer::initiate(const std::string &csvFileName, std::ofstream &csvFileStream, const std::string &inputVideo){
    inputVideoName = inputVideo;
    inputVideoName = inputVideoName.substr(inputVideoName.find_last_of("/"));
    inputVideoName = inputVideoName.substr(0, inputVideoName.size()-4);
    std::cout << inputVideoName << std::endl;
    auto timed = std::chrono::system_clock::now();
    //std::time_t currentTime = std::chrono::system_clock::to_time_t(timed);
    myFileName = csvFileName;
    myFileName = myFileName.append(inputVideoName+"_det.txt");
    csvFileStream.open(myFileName);
}