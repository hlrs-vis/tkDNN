#include "VideoAcquisition.h"
#include <thread>
#include <chrono>

VideoAcquisition::VideoAcquisition()
{

}

void VideoAcquisition::getImages(std::vector<TypewithMetadata<cv::Mat>> *batch_images, int n_batch)
{
    int num_elems_removed = 0;
    while (m_queue.size() < n_batch )
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (m_queue.size() > m_max_number_queued+(n_batch))
    {
        int num_elems_to_get = m_queue.size()-m_max_number_queued;
        //std::cout << num_elems_to_get << " elements will be removed\n";
        int elem = 0;
        for (int bi = 0; bi < n_batch; ++bi)
        {
            while (elem < int((double(num_elems_to_get))/n_batch*bi))
            {
                m_queue.pop_front();
                //std::cout << "Omitted Element " << elem << "\n";
                elem++;
                num_elems_removed++;
            }
            
            {
            batch_images->push_back(std::move(m_queue.front()));
            m_queue.pop_front();
            //std::cout << "Got Element " << elem << "\n";
            elem++;
            }
        }
        while (elem<num_elems_to_get)
        {
            m_queue.pop_front();
            elem++;
            num_elems_removed++;
            //std::cout << "Omitted Element " << elem << "\n";
        }
        // m_queue.pop_front();
    }
    else
    {
        for (int bi = 0; bi < n_batch; ++bi)
        {
            batch_images->push_back(std::move(m_queue.front()));
            m_queue.pop_front();
        }
    }
    // std::cout << n_batch << " images retrieved, " << num_elems_removed << " removed, queue size is " << m_queue.size()<< "\n";
}

void VideoAcquisition::flip()
{
    m_flipped = true;
}

void VideoAcquisition::setPlayback()
{
    m_playback = true;
}

void VideoAcquisition::setAdjustExposure()
{
    m_adjust_exposure = true;
}

void VideoAcquisition::calculateMean(TypewithMetadata<cv::Mat> &frame)
{
    if (m_num_mean_values == 0)
    {
        m_mean = cv::mean(frame.data);
        m_num_mean_values +=1;
    }
    else
    {
        m_mean += cv::mean(frame.data);
        m_num_mean_values +=1;
    }
    cv::Scalar m_mean_average = m_mean/m_num_mean_values;
    /*std::cout << "Frame mean" << m_mean_average << std::endl;
    std::cout << "Number of Frames" << m_num_mean_values << std::endl;*/
    double mean_array[3] = {m_mean_average[0], m_mean_average[1], m_mean_average[2] };
    max_mean_value = *std::max_element(mean_array, mean_array+3);
    //std::cout << "Max" << max_mean_value << std::endl;
 }