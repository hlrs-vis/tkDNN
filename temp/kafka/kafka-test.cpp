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

    cv::Mat image = readImage("../cow.png");
    
    if (!image.empty()) {
        cv::imshow("Image", image);
        cv::waitKey(0);  
        cv::destroyAllWindows();   
    } else {
        std::cerr << "Failed to load image!" << std::endl;
    }

    // int partition = 0;
    // string topic = "timed-images";
    // KafkaProducer kafka_producer("localhost:9092");
    // string timestamp = getTime();
    // json frame;
    // frame["frame_id"] = "1";
    // frame["cam_id"] =  "1";
    // frame["timestamp"] = timestamp;
    // for (int i = 0; i < 3; ++i) {
    //     json detection;
    //     detection["detection_class"] = "class_" + std::to_string(i);
    //     detection["bbox_x"] = "x_" + std::to_string(i);
    //     detection["bbox_y"] = "y_" + std::to_string(i);
    //     detection["bbox_w"] = "w_" + std::to_string(i);
    //     detection["bbox_h"] = "h_" + std::to_string(i);
    //     detection["probability"] = "prob_" + std::to_string(i);
    //     json featuresArray = {"feature1" + std::to_string(i), "feature2" + std::to_string(i), "feature3" + std::to_string(i)};
    //     detection["features"] = featuresArray;
    //     frame["detections"].push_back(detection);
    // }


    // string serialized_message = frame.dump();
    // kafka_producer.produceMessage(topic, serialized_message, partition);
    
    
}

string getTime() {

    auto currentTime = std::chrono::system_clock::now();
    std::time_t currentTimeT = std::chrono::system_clock::to_time_t(currentTime);
    auto duration = currentTime.time_since_epoch();

    auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
    duration -= hours;
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
    duration -= minutes;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
    duration -= seconds;
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);

    std::string timeString =
        std::to_string(hours.count()) + ";" +
        std::to_string(minutes.count()) + ";" +
        std::to_string(seconds.count()) + ";" +
        std::to_string(milliseconds.count());

    return timeString;
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
