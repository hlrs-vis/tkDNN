
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
    virtual int getWidth() = 0;
    virtual int getHeight() = 0;
    void getImages(std::vector<TypewithMetadata<cv::Mat>> *batch_images, int n_batch);
    virtual void setFrameRate(int) = 0;
    void flip();
    void setPlayback();
    void setAdjustExposure();
    void calculateMean(TypewithMetadata<cv::Mat> &frame);
    void adjustExposure();
    virtual void setExposure(int exposure = 0) = 0;

    void set_exposure_adjust_interval(int exposure_adjust_interval);
    void set_exposure_adjust_step(int m_exposure_adjust_step);
    void set_exposure_max_desired_mean_value (int exposure_max_desired_mean_value );
    void set_exposure_min_desired_mean_value(int exposure_min_desired_mean_value);
    void set_exposure_min(int exposure_min);
    void set_exposure_max(int exposure_max);



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
    bool m_adjust_exposure = 0;
    bool m_calculate_background_image = 0;
    cv::Scalar m_mean;
    int m_num_mean_values = 0;
    int m_exposure_adjust_interval = 30;
    int m_exposure_adjust_step = 3;
    int m_exposure_max_desired_mean_value = 70;
    int m_exposure_min_desired_mean_value = 30;
    int m_exposure = 50;
    int m_exposure_min = 3;
    int m_exposure_max = 2000;
    double max_mean_value = 0;

private:



};


#endif /* VideoAcquisition_H*/