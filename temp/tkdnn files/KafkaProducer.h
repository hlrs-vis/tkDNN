#ifndef KAFKA_PRODUCER_H
#define KAFKA_PRODUCER_H

#include <nlohmann/json.hpp>
#include <cppkafka/cppkafka.h>
#include <DetectionWithFeatureVector.h>
#include <vector>
#include <iostream>

using json = nlohmann::json;
using namespace std;
using namespace cppkafka;

class KafkaProducer {
    public:
        KafkaProducer(const string& broker_list);
        ~KafkaProducer();

        void produceMessage(const string& topic, const string& message, const int& partition);
        vector<json> turnDetectionsToJson(const vector<vector<DetectionWithFeatureVector>>& detections, const int& cam_id);

    private:
        Configuration config;
        Producer producer;
};

#endif