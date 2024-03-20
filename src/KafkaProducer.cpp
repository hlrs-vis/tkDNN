#include "KafkaProducer.h"

KafkaProducer::KafkaProducer(const string& broker_list) : config{{"metadata.broker.list", broker_list}}, producer(config) {
    Configuration config = {
          { "metadata.broker.list", broker_list }
    };  
    
    Producer producer(config);
}

KafkaProducer::~KafkaProducer() {
    // Ensure any outstanding messages are delivered
    producer.flush(1000ms); 
}

void KafkaProducer::produceMessage(const string& topic, string message, int partition) {

    // Produce the message
    try {

        producer.produce(MessageBuilder(topic).partition(partition).payload(message));
    } catch (const cppkafka::HandleException& e) {
        std::cerr << "Error:" << e.what() << std::endl;
    }
    producer.flush();

}

std::vector<json> KafkaProducer::turnDetectionsToJson(const std::vector<std::vector<DetectionWithFeatureVector>>& batch_detections){

    // Create empty array to hold the detections
    std::vector<json> jsonDetections;

    // Loop over all frames
    for (int bi = 0; bi < batch_detections.size(); ++bi){
        // Loop over all detections
        json singleFrameDetections = json::array();
        for (int i = 0; i < batch_detections[bi].size(); ++i){
            json features = json::array();
            // Loop over all entries in feature_vector
            for (int j = 0; j < batch_detections[bi][i].feature_vector.size(); j++) { // Save all entries of the feature vector as strings in JSON format
                json feature = {
                    std::to_string(batch_detections[bi][i].feature_vector[j])
                };
                features.push_back(feature);
            }
            json singleDetection = {
                { "f", std::to_string(batch_detections[bi][i].frame_id)},
                { "bX", std::to_string(batch_detections[bi][i].bbox_x)},
                { "bY", std::to_string(batch_detections[bi][i].bbox_y)},
                { "bW", std::to_string(batch_detections[bi][i].bbox_w)},
                { "bH", std::to_string(batch_detections[bi][i].bbox_h)},
                { "p", std::to_string(batch_detections[bi][i].probability)},
                { "features", features}
            };
            singleFrameDetections.push_back(singleDetection);
        }
        jsonDetections.push_back(singleFrameDetections);
    }
    return jsonDetections;
}