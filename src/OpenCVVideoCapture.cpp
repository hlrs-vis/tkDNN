#include "OpenCVVideoCapture.h"

OpenCVVideoCapture::OpenCVVideoCapture()
{

}

bool OpenCVVideoCapture::init(std::string input, int video_mode)
{
    cap.open(input);
    if (!cap.isOpened())
    {
        m_isOpened = false;
    }
    else
    {
        std::cout << "camera started\n";
        cap.set(cv::CAP_PROP_FOURCC,cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 1);

        cap.set(cv::CAP_PROP_EXPOSURE, m_exposure);
        switch (video_mode)
        {
            case 0:
                cap.set(cv::CAP_PROP_FRAME_WIDTH,1920);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT,1080);
                cap.set(cv::CAP_PROP_FPS,60);
                break;
            case 1:
                cap.set(cv::CAP_PROP_FRAME_WIDTH,3840);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT,2160);
                cap.set(cv::CAP_PROP_FPS,30);
                break;
            default:
                cap.set(cv::CAP_PROP_FRAME_WIDTH,1920);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT,1080);
                cap.set(cv::CAP_PROP_FPS,60);

        }
        cap.set(cv::CAP_PROP_FPS,60);
        int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        std::cout << "Width: " << w << " Height: " << h << "\n";
        cv::String outFileName = "calibrationFrame" + std::to_string(0);
        outFileName.append(".jpg");
        cv::Mat calibrationFrame;
        cap >> calibrationFrame;
        cv::imwrite(outFileName,calibrationFrame);
    }
    return 0;
}

int OpenCVVideoCapture::getHeight()
{       
    if (cap.isOpened())
        m_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    return m_height;
}

int OpenCVVideoCapture::getWidth()
{
    if (cap.isOpened())
        m_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    return m_width;
}

void OpenCVVideoCapture::acquisition_thread()
{
    TypewithMetadata<cv::Mat> newframe;

    while (m_isRunning)
    {
        if (m_playback)
        {
            while (m_queue.size() >= m_max_number_queued)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        cap >> newframe.data;
        if (m_flipped)
            cv::flip(newframe.data, newframe.data, -1);
        newframe.time = std::chrono::system_clock::now();
        newframe.frame_id = m_frame_id;
        m_frame_id++;
        if (m_adjust_exposure)
        {
            calculateMean(newframe);
            if (m_num_mean_values >= m_exposure_adjust_interval)
            {
                adjustExposure();
            }
        }
        m_queue.push_back(std::move(newframe));
    }
}

void OpenCVVideoCapture::start()
{
    m_isRunning = true;
    std::thread camera_thread(&OpenCVVideoCapture::acquisition_thread, this);
    camera_thread.detach();
}

 void OpenCVVideoCapture::stop()
{
    m_isRunning = false;
    std::cout << "Frames dropped: " << m_frames_dropped << "\n";
}

/* bool OpenCVVideoCapture::init()
{
}
*/

void OpenCVVideoCapture::setFrameRate(int fps)
{
    cap.set(cv::CAP_PROP_FPS,fps);
}


void OpenCVVideoCapture::adjustExposure()
{
    if (max_mean_value > m_exposure_max_desired_mean_value)
    {
        m_exposure -= 1;
        cap.set(cv::CAP_PROP_EXPOSURE, m_exposure);
    }
    if (max_mean_value > m_exposure_min_desired_mean_value)
    {
        m_exposure += 1;
        cap.set(cv::CAP_PROP_EXPOSURE, m_exposure);
    }
}

void OpenCVVideoCapture::adjustExposure()
{
    if (max_mean_value > m_exposure_max_desired_mean_value)
    {
        m_exposure -= 1;
        cap.set(cv::CAP_PROP_EXPOSURE, m_exposure);
        if (m_exposure < m_exposure_min)
            m_exposure = m_exposure_min;
    }
    else if (max_mean_value < m_exposure_min_desired_mean_value)
    {
        m_exposure += 1;
        cap.set(cv::CAP_PROP_EXPOSURE, m_exposure);
        if (m_exposure > m_exposure_max)
            m_exposure = m_exposure_max;
    }
    else
    {
        std::cout << " Nothing to do" << std::endl;
    }
    std::cout << "Exposure is: " << m_exposure << std::endl;
    m_num_mean_values = 0;
}