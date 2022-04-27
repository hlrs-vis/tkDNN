#include "IDSVideoCapture.h"

IDSVideoCapture::IDSVideoCapture()
{

}

bool IDSVideoCapture::init(std::string input, int video_mode)
{
    if (!IDSCam.isRunning())
    {
        m_isOpened = false;
    }
    else
    {
        std::cout << "camera started\n";
        IDSCam.setFrameRate(30);
    }
    return 0;
}

int IDSVideoCapture::getHeight()
{       
    if (IDSCam.isRunning())
        m_height = IDSCam.getHeight();
    return m_height;
}

int IDSVideoCapture::getWidth()
{
    if (IDSCam.isRunning())
        m_width = IDSCam.getWidth();
    return m_width;
}

void IDSVideoCapture::acquisition_thread()
{
    while (m_isRunning)
    {
        TypewithMetadata<cv::Mat> newframe;
        newframe.data = IDSCam.getFrame();
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

void IDSVideoCapture::start()
{
    m_isRunning = true;
    std::thread camera_thread(&IDSVideoCapture::acquisition_thread, this);
    camera_thread.detach();
}

 void IDSVideoCapture::stop()
{
    m_isRunning = false;
    std::cout << "Frames dropped: " << m_frames_dropped << "\n";
}

/* bool IDSVideoCapture::init()
{
}
*/

void IDSVideoCapture::setFrameRate(int fps)
{
       IDSCam.setFrameRate(fps);
}

void IDSVideoCapture::adjustExposure()
{
    if (max_mean_value > m_exposure_max_desired_mean_value)
    {
        m_exposure -= 1;
        std::cout << " Exposure adjustion not implemented for IDS" << std::endl;
        if (m_exposure < m_exposure_min)
            m_exposure = m_exposure_min;
    }
    else if (max_mean_value < m_exposure_min_desired_mean_value)
    {
        m_exposure += 1;
        std::cout << " Exposure adjustion not implemented for IDS " <<  std::endl;
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