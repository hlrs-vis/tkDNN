
#ifndef VideoAcquisition_H
#define VideoAcquisition_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include "opencv2/opencv.hpp"

#include <chrono>
#include <thread>

#include "SharedQueue.h"
#include "TypewithMetadata.h"

class VideoAcquisition
{

public:
    VideoAcquisition();
    
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual bool init(std::string, int) = 0;
    void getImages(std::vector<TypewithMetadata<cv::Mat>> *batch_images, int n_batch);
    virtual void setFrameRate(int) = 0;
    void flip();
    void setPlayback();

protected: 
    bool m_isOpened = false;
    bool m_isRunning = false;
    int m_width = 0;
    int m_height = 0;
    SharedQueue<TypewithMetadata<cv::Mat>> m_queue;
    long long int m_frame_id = 0;
    int m_max_number_queued = 100;
    int m_frames_dropped = 0;
    bool m_flipped = 0;
    bool m_playback = 0;

private:



};


#endif /* VideoAcquisition_H*/