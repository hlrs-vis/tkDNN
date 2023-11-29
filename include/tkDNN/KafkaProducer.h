#ifndef KAFKA_PRODUCER_H
#define KAFKA_PRODUCER_H

#include <nlohmann/json.hpp>
#include <cppkafka/cppkafka.h>
#include "TypewithMetadata.h"

class KafkaProducer {
    public:
        KafkaProducer(const std::string& broker_list);
        ~KafkaProducer();

        void produceMessage(const std::string& topic, json message, int partition = 0);
        json turnDetectionsToJson(const vector<std::float>& detections, std::vector<TypewithMetadata<cv::Mat>> *batch_images);

    private:
        cppkafka::Configuration config;
        cppkafka::Producer producer;
};
