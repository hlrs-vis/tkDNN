#ifndef KAFKA_PRODUCER_H
#define KAFKA_PRODUCER_H

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <cppkafka/cppkafka.h>
#include "TypewithMetadata.h"
#include <DetectionWithFeatureVector.h>
#include <vector>

using json = nlohmann::json;
using namespace std;
using namespace cppkafka;

class KafkaProducer {
    public:
        KafkaProducer(const string& broker_list);
        ~KafkaProducer();

        void produceMessage(const string& topic, string message, int partition = 0);
        vector<json> turnDetectionsToJson(const vector<vector<DetectionWithFeatureVector>>& detections);

    private:
        Configuration config;
        Producer producer;
};


#endif