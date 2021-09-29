#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include "opencv2/opencv.hpp"

#include "VideoAcquisition.h"


// class VideoAcquisition;

class OpenCVVideoCapture : public VideoAcquisition
{

//friend class VideoAcquisition;

public:
    OpenCVVideoCapture();

    void start();
    void stop();
    bool init(std::string input="/dev/video1", int video_mode = 1);
    void setFrameRate(int fps);
    //bool init();
    int getWidth();
    int getHeight();

private:
cv::VideoCapture cap;
void acquisition_thread();
//bool m_isOpened = false;

};