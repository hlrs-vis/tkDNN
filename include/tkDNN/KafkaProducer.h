#ifndef KAFKA_PRODUCER_H
#define KAFKA_PRODUCER_H

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <cppkafka/cppkafka.h>
#include "TypewithMetadata.h"
#include <DetectionWithFeatureVector.h>
#include <vector>

using json = nlohmann::json;
using namespace cppkafka;

class KafkaProducer {
    public:
        KafkaProducer(const std::string& broker_list);
        ~KafkaProducer();

        void produceMessage(const std::string& topic, json message, int partition = 0);
        json turnDetectionsToJson(const std::vector<DetectionWithFeatureVector>& detections, std::vector<TypewithMetadata<cv::Mat>> *batch_images);

    private:
        cppkafka::Configuration config;
        cppkafka::Producer producer;
};


#endif