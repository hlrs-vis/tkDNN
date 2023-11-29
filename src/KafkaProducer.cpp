#include "KafkaProducer.h"

KafkaProducer::KafkaProducer(const std::string& broker_list)
    : config{
          { "metadata.broker.list", broker_list }
      },
      producer(config) {}

KafkaProducer::~KafkaProducer() {
    producer.flush(); // Ensure any outstanding messages are delivered
}

void KafkaProducer::produceMessage(const std::string& topic, int partition) {
    cppkafka::MessageBuilder builder(topic);
    // Construct metadata for the message, each message consists of a single frame plus all the detections on this frame
    
    
    
    
    builder.partition(partition).payload(jsonDetections);
    // Produce the message
    producer.produce(builder);

}

json KafkaProducer::turnDetectionsToJson(const vector<DetectionWithFeatureVector>& detections, std::vector<TypewithMetadata<cv::Mat>> *batch_images){

    // Create empty array to hold the detections
    json jsonDetections = json::array();

    // Loop over all images, create the metadata, detections 
    for (int bi = 0; bi < batch_images.size(); ++bi){
        for (const auto& detection : detections){
            
            if (detection.frame_id == bi){  // Save only these detections, that match the frame
                
                json features = json::array();
                for (int i = 0; i < detection.feature_vector.size(); i++) { // Save all entries of the feature vector as strings in JSON format
                    json feature = {
                        std::to_string(detection.feature_vector[i])
                    }
                    features.push_back(feature)
                }
                json detection = {
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
            jsonDetections.push_back(detection);
        }

    }
}