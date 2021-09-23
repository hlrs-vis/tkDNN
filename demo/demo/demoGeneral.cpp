#include <iostream>
#include <signal.h>
#include <stdlib.h> /* srand, rand */
#include <unistd.h>
#include <mutex>
#include <https_stream.h> //https_stream

#include <chrono>
#include <ctime>
#include <time.h>

#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo3Detection.h"

#include "SharedQueue.h"
#include "TypewithMetadata.h"
#include "OpenCVVideoCapture.h"
#include "IDSVideoCapture.h"

bool gRun;
bool SAVE_RESULT = false;

void sig_handler(int signo)
{
    std::cout << "request gateway stop\n";
    gRun = false;
}

int main(int argc, char *argv[])
{

    std::cout << "detection\n";
    signal(SIGINT, sig_handler);
    
    // JSON-Port
    int json_port = find_int_arg(argc, argv, "-json_port", -1);
  
    // Net
    char *inputnet = find_char_arg(argc, argv, "-net", "yolo3_berkeley.rt");
    std::string net(inputnet);

    // Input 
    char *inputvideo = find_char_arg(argc, argv, "-input", "../demo/yolo_test.mp4");
    std::string input(inputvideo); 

    //Net-Type  
    char *input_ntype = find_char_arg(argc, argv, "-ntype", "y");
    char ntype = input_ntype[0];

    int n_classes = find_int_arg(argc, argv, "-n_classes", 80);
    
    int n_batch = find_int_arg(argc, argv, "-n_batch", 1);

    int show = find_int_arg(argc, argv, "-show", 1);

    int save = find_int_arg(argc, argv, "-save", 0);
    SAVE_RESULT = save;

    int ids = find_int_arg(argc, argv, "-ids", 0);

    int mjpeg_port = find_int_arg(argc, argv, "-mjpeg_port", 0);
  
    int extyolo = find_int_arg(argc, argv, "-extyolo", 0);

    int video_mode = find_int_arg(argc, argv, "-video_mode", 0);

    if (n_batch < 1 || n_batch > 64)
        FatalError("Batch dim not supported");


    tk::dnn::Yolo3Detection yolo;
    tk::dnn::CenternetDetection cnet;
    tk::dnn::MobilenetDetection mbnet;

    tk::dnn::DetectionNN *detNN;

    switch (ntype)
    {
    case 'y':
        detNN = &yolo;
        break;
    case 'c':
        detNN = &cnet;
        break;
    case 'm':
        detNN = &mbnet;
        n_classes++;
        break;
    default:
        FatalError("Network type not allowed (3rd parameter)\n");
    }

    detNN->init(net, n_classes, n_batch);

    gRun = true;

bool draw = (show || SAVE_RESULT);

VideoAcquisition *video;

if (!ids){
    video = new OpenCVVideoCapture;
}
else
{
    video = new IDSVideoCapture;
}

video->init(input, video_mode);

video->start();

    cv::VideoWriter resultVideo;
    if (SAVE_RESULT)
    {   /*
        int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        resultVideo.open("result.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, cv::Size(w, h));
        */
    }

    cv::Mat frame;
    if (show)
    {
        cv::namedWindow("detection", cv::WINDOW_NORMAL);
        cv::moveWindow("detection", 100, 100);
        cv::resizeWindow("detection", 1920, 1080);
    }

    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;

    std::vector<TypewithMetadata<cv::Mat>> *batch_images = new std::vector<TypewithMetadata<cv::Mat>>;
    TypewithMetadata<cv::Mat> image;

    std::chrono::time_point<std::chrono::system_clock> start_time = std::chrono::system_clock::now();
    int frames_processed = 0;
    while (gRun)
    {
        //ensure queue holds enough pictures for batch size


        batch_dnn_input.clear();
        batch_frame.clear();
        batch_images->clear();

        video->getImages(batch_images, n_batch);
        
        for (int bi = 0; bi < n_batch; ++bi)
        {
            // this will be used for the visualisation
            if (draw)
                batch_frame.push_back((*batch_images)[bi].data);

            // this will be resized to the net format
            if (!draw)
                //batch_dnn_input.push_back((*batch_images)[bi].data);
                batch_dnn_input.push_back(std::move((*batch_images)[bi].data));
            else
            batch_dnn_input.push_back((*batch_images)[bi].data);
        }

        //inference
        detNN->update(batch_dnn_input, n_batch);
        if (draw)
            detNN->draw(batch_frame,extyolo);
        frames_processed += n_batch;

        if (show)
        {
            for (int bi = 0; bi < n_batch; ++bi)
            {
                cv::imshow("detection", batch_frame[bi]);
            }
            //cv::imshow("detection", batch_frame[0]);
        }
        if (cv::waitKey(1) == 27)
        {
            break;
        }
        if (n_batch == 1 && SAVE_RESULT)
            resultVideo << batch_frame[0];

        if (mjpeg_port > 0)
        {
            send_mjpeg(batch_frame[0], mjpeg_port, 400000, 40);
        }
        
        if (json_port > 0)
        {
            send_json(batch_images, *detNN, json_port, 40000);
        }
    }

    video->stop();
    long long int frame_id = (*batch_images)[n_batch-1].frame_id;

    std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();

    std::cout << "detection end\n";
    double mean = 0;

    std::cout << COL_GREENB << "\n\nTime stats:\n";
    std::cout << "Min: " << *std::min_element(detNN->stats.begin(), detNN->stats.end()) / n_batch << " ms\n";
    std::cout << "Max: " << *std::max_element(detNN->stats.begin(), detNN->stats.end()) / n_batch << " ms\n";
    for (int i = 0; i < detNN->stats.size(); i++)
        mean += detNN->stats[i];
    mean /= detNN->stats.size();
    std::cout << "Avg: " << mean / n_batch << " ms\t" << 1000 / (mean / n_batch) << " FPS\n"
              << COL_END;

    std::cout << COL_GREENB << "Frames overall: " << frames_processed / std::chrono::duration<double>(end_time-start_time).count() << " fps \n" << COL_END;

    return 0;
}
