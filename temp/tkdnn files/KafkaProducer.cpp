#include "KafkaProducer.h"

KafkaProducer::KafkaProducer(const string& broker_list) : config{{"metadata.broker.list", broker_list}}, producer(config) {
    Configuration config = {
          { "metadata.broker.list", broker_list }
    };  
    std::cerr << "Producer constructor"<< std::endl;
    produceMessage("timed-images", "test", 0);
}

KafkaProducer::~KafkaProducer() {
    // Ensure any outstanding messages are delivered
    producer.flush(1000ms); 
}

void KafkaProducer::produceMessage(const string& topic, const string& message, const int& partition) {
    // Produce the message
    try {
        producer.produce(MessageBuilder(topic).partition(partition).payload(message));
    } 
    catch (const cppkafka::Exception& e) {
    std::cerr << "Kafka exception: " << e.what() << std::endl;
        try {
            // Attempt to rethrow if nested exceptions are supported
            std::rethrow_if_nested(e);
        } catch(const std::exception& ne) {
            // Handle nested exception
            std::cerr << "Nested exception: " << ne.what() << std::endl;
        } catch(...) {
            std::cerr << "An unknown nested exception occurred." << std::endl;
        }
    }
}

std::vector<json> KafkaProducer::turnDetectionsToJson(const std::vector<std::vector<DetectionWithFeatureVector>>& batch_detections, const int& cam_id){

    // Create empty array to hold the detections
    std::vector<json> jsonDetections;

    // Loop over all frames
    for (int bi = 0; bi < batch_detections.size(); ++bi){
        // Loop over all detections
        json singleFrameDetections = json::array();
        singleFrameDetections["cam_id"] = 1;
        singleFrameDetections["frame_id"] = batch_detections[bi][0].frame_id; // Just use the first detection to get the frame_id, as they are all identical
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