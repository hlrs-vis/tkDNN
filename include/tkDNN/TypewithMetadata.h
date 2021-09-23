#ifndef TYPEWITHMETADATA_H
#define TYPEWITHMETADATA_H

#include <chrono>

template<typename T>
struct TypewithMetadata {
    T data;
    std::chrono::time_point<std::chrono::system_clock> time;
    long long int frame_id;
    /*TypewithMetadata(T image)
    {
        data =image;
        time = std::chrono::system_clock::now();
    }*/
};

#endif /* TYPEWITHMETADATA_H */