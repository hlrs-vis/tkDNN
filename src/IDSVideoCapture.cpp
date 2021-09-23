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
        IDSCam.setFrameRate(50);
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
        newframe.time = std::chrono::system_clock::now();
        newframe.frame_id = m_frame_id;
        m_frame_id++;
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