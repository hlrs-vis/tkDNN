cmake_minimum_required(VERSION 3.10)
project(KafkaTest)

# add execuable
add_executable(KafkaTest kafka-test.cpp)

# add source files
file(GLOB kafkaTest_src "src/*.cpp")
add_library(kafkaTest_src SHARED ${kafkaTest_src})

include_directories(/usr/local/src/git/tkDNN/temp/kafka/include/kafkaTest)
include_directories(/usr/local/src/)

find_package(OpenCV REQUIRED)
find_package(CppKafka REQUIRED)

# JSON
set(CMAKE_PREFIX_PATH "/usr/local/src/git/json/build")
find_package(nlohmann_json 3.2.0 REQUIRED HINTS "/usr/local/src/git/json/build")
target_link_libraries(KafkaTest nlohmann_json::nlohmann_json)

target_link_libraries(KafkaTest kafkaTest_src CppKafka::cppkafka ${OpenCV_LIBS})
