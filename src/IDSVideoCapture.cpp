#include "IDSVideoCapture.h"

IDSVideoCapture::IDSVideoCapture()
{

}

bool IDSVideoCapture::init(std::string input, int video_mode)
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
        switch (video_mode)
        {
            case 0:
                cap.set(cv::CAP_PROP_FRAME_WIDTH,1920);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT,1080);
                break;
            case 1:
                cap.set(cv::CAP_PROP_FRAME_WIDTH,3840);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT,2160);
                break;
            default:
                cap.set(cv::CAP_PROP_FRAME_WIDTH,1920);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT,1080);

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

int IDSVideoCapture::getHeight()
{       
    if (cap.isOpened())
        m_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    return m_height;
}

int IDSVideoCapture::getWidth()
{
    if (cap.isOpened())
        m_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    return m_width;
}

void IDSVideoCapture::acquisition_thread()
{
    while (m_isRunning)
    {
        while (m_queue.size() > m_max_number_queued)
        {
            m_queue.pop_front();
            m_frames_dropped++;
        }
        TypewithMetadata<cv::Mat> newframe;
        cap >> newframe.data;
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