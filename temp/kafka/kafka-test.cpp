#include <cppkafka/cppkafka.h>
#include "KafkaProducer.h"
#include <chrono>
#include <ctime>
#include <time.h>
#include "opencv2/opencv.hpp"
#include <nlohmann/json.hpp>


using namespace std;
using namespace cppkafka;
using json = nlohmann::json;

cv::Mat readImage(const string& filename);
string getTime();

int main() {
    KafkaProducer* kafkaProducer = nullptr;
    if (true) {
        kafkaProducer = new KafkaProducer("localhost:9092");
        std::cout << "initialized"  << std::endl;
    }
    int partition = 0;
    string topic = "timed-images";
    string timestamp = getTime();
    json frame;
    frame["frame_id"] = "1";
    frame["cam_id"] =  "1";
    frame["timestamp"] = timestamp;
    frame["detections"] = nullptr;
    for (int i = 0; i < 3; ++i) {
        json detection;
        detection["bbox_x"] = std::to_string(i);
        detection["bbox_y"] = std::to_string(i);
        detection["bbox_w"] = std::to_string(i);
        detection["bbox_h"] = std::to_string(i);
        detection["probability"] = std::to_string(i);
        json featuresArray = {std::to_string(i), std::to_string(i), std::to_string(i)};
        detection["features"] = featuresArray;
        frame["detections"].push_back(detection);
    }

    string serialized_message = frame.dump();
    if (kafkaProducer) {
        kafkaProducer->produceMessage(topic, serialized_message, partition);
        std::cout << "produced?"  << std::endl;
    }
    delete kafkaProducer;
}

string getTime() {

    auto currentTime = std::chrono::system_clock::now();
    std::time_t currentTimeT = std::chrono::system_clock::to_time_t(currentTime);
    auto duration = currentTime.time_since_epoch();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);

    std::string timeString = std::to_string(milliseconds.count());

    return timeString;
}

