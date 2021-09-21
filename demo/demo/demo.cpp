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

    cv::VideoCapture cap(input);
    if (!cap.isOpened())
    {
        gRun = false;
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
        int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        std::cout << "Width: " << w << " Height: " << h << "\n";
        cv::String outFileName = "calibrationFrame" + std::to_string(0);
        outFileName.append(".jpg");
        cv::Mat calibrationFrame;
        cap >> calibrationFrame;
        cv::imwrite(outFileName,calibrationFrame);
    }

    cv::VideoWriter resultVideo;
    if (SAVE_RESULT)
    {
        int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        resultVideo.open("result.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, cv::Size(w, h));
    }

    cv::Mat frame;
    if (show)
        cv::namedWindow("detection", cv::WINDOW_NORMAL);
    cv::moveWindow("detection", 100, 100);
    cv::resizeWindow("detection", 1920, 1080);

    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;

    std::vector<std::chrono::time_point<std::chrono::system_clock>> batch_frame_time;

    std::vector<long long int> frame_ids;
    long long int frame_id = 0;
    std::chrono::time_point<std::chrono::system_clock> start_time = std::chrono::system_clock::now();

    while (gRun)
    {
        std::clock_t c_start = std::clock();
        auto t_start = std::chrono::high_resolution_clock::now();
        batch_dnn_input.clear();
        batch_frame.clear();
        frame_ids.clear();
        batch_frame_time.clear();

        for (int bi = 0; bi < n_batch; ++bi)
        {
            cap >> frame;
            if (!frame.data)
                break;

            batch_frame_time.push_back(std::chrono::system_clock::now());
            frame_ids.push_back(frame_id);
            frame_id++;
            batch_frame.push_back(frame);

            // this will be resized to the net format
            batch_dnn_input.push_back(frame.clone());
        }
        if (!frame.data)
            break;

        //inference
        detNN->update(batch_dnn_input, n_batch);
        detNN->draw(batch_frame,extyolo);

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
            resultVideo << frame;

        if (mjpeg_port > 0)
        {
            send_mjpeg(batch_frame[0], mjpeg_port, 400000, 40);
        }
        
        if (json_port > 0)
        {
            send_json(batch_frame, *detNN, frame_ids, json_port, 40000, &batch_frame_time);
        }
        /*
        std::clock_t c_end = std::clock();
        auto t_end = std::chrono::high_resolution_clock::now();
        auto time_lowres = std::chrono::system_clock::now();


        std::cout << std::fixed << std::setprecision(4) << "CPU time used: "
        << 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC << " ms\n"
        << "Wall clock time passed: "
        << std::chrono::duration<double, std::milli>(t_end-t_start).count()
        << " ms\n" << "Highres Time is:" << t_start.time_since_epoch().count() << "\n"
        << " ms\n" << "Lowres Time is:" << std::chrono::duration<double, std::milli>(time_lowres.time_since_epoch()).count() << "\n";*/
    }

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

    std::cout << COL_GREENB << "Frames overall: " << frame_id / std::chrono::duration<double>(end_time-start_time).count() << " fps \n" << COL_END;

    return 0;
}
