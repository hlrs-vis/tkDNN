#include "KafkaProducer.h"

KafkaProducer::KafkaProducer(const string& broker_list) : config{{"metadata.broker.list", broker_list}}, producer(config) {
    Configuration config = {
          { "metadata.broker.list", broker_list }
    };  
    
    Producer producer(config);
}

KafkaProducer::~KafkaProducer() {
    // Ensure any outstanding messages are delivered
    producer.flush(); 
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

string KafkaProducer::turnDetectionsToJson(const vector<DetectionWithFeatureVector>& detections, vector<TypewithMetadata<cv::Mat>> *batch_images){

    // Create empty array to hold the detections
    json jsonDetections = json::array();

    // Loop over all images, create the metadata, detections 
    for (int bi = 0; bi < batch_images->size(); ++bi){
        for (const auto& detection : detections){
            
            if (detection.frame_id == bi){  // Save only these detections, that match the frame
                
                json features = json::array();
                for (int i = 0; i < detection.feature_vector.size(); i++) { // Save all entries of the feature vector as strings in JSON format
                    json feature = {
                        std::to_string(detection.feature_vector[i])
                    };
                    features.push_back(feature);
                }
                json singleDetection = {
                    { "f", std::to_string(detection.frame_id)},
                    { "c", std::to_string(detection.detection_class)},
                    { "bX", std::to_string(detection.bbox_x)},
                    { "bY", std::to_string(detection.bbox_y)},
                    { "bW", std::to_string(detection.bbox_w)},
                    { "bH", std::to_string(detection.bbox_h)},
                    { "p", std::to_string(detection.probability)},
                    { "gX", std::to_string(detection.global_x)},
                    { "gY", std::to_string(detection.global_y)},
                    { "gZ", std::to_string(detection.global_z)},
                    { "features", features}
                };
                jsonDetections.push_back(singleDetection);
            }
        }

    }
    return jsonDetections.dump();
}