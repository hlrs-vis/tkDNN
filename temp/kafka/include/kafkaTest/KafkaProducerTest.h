#ifndef KAFKA_PRODUCER_H
#define KAFKA_PRODUCER_H

#include <cppkafka/cppkafka.h>
#include <vector>
#include <iostream>

using namespace std;
using namespace cppkafka;

class KafkaProducer {
    public:
        KafkaProducer(const string& broker_list);
        ~KafkaProducer();

        void produceMessage(const string& topic, const string& message, const int& partition);
        

    private:
        Configuration config;
        Producer producer;
};


#endif