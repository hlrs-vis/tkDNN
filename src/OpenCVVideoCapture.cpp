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
        cap >> newframe.data;
        newframe.time = std::chrono::system_clock::now();
        newframe.frame_id = m_frame_id;
        m_frame_id++;
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