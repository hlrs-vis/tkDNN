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

string encodeImage(const cv::Mat& image);

int main() {

    cv::Mat image = readImage("../cow.png");
    


    int partition = 0;
    string topic = "timed-images";
    KafkaProducer kafka_producer("localhost:9092");
    json message;
    message["str1"] = "test";
    message["str2"] = "message";
    string serialized_message = message.dump();
    kafka_producer.produceMessage(topic, serialized_message, partition);
    
    
}

cv::Mat readImage(const string& filename) {
    
    try {
        cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw cv::Exception(0, "Image not loaded", "main", "Your file may not exist or be a valid image file",-1);
        }
        else {
            return image;
        }
    } catch (const cv::Exception& ex) {
        std::cerr << "OpenCV Exception: " << ex.what() << std::endl;
        std::cerr << "Failed to load image, creating black one" << std::endl;
        cv::Mat image(1080, 1920, CV_8UC3, cv::Scalar(0,0,0));

        return image;
    }
}
