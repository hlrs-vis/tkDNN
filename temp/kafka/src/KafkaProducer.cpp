#include "KafkaProducer.h"

KafkaProducer::KafkaProducer(const string& broker_list) : config{{"metadata.broker.list", broker_list}}, producer(config) {
    Configuration config = {
          { "metadata.broker.list", broker_list }
    };  
    
}

KafkaProducer::~KafkaProducer() {
    // Ensure any outstanding messages are delivered
    producer.flush(3000ms); 
}

void KafkaProducer::produceMessage(const string& topic, const string& message, const int& partition) {

    // Produce the message
    try {

        producer.produce(MessageBuilder(topic).partition(partition).payload(message));
    } catch (const cppkafka::HandleException& e) {
        std::cerr << "Error:" << e.what() << std::endl;
    }
}
