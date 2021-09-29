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


#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/xml_parser.hpp>

bool gRun;
bool SAVE_RESULT = false;

using namespace boost::property_tree;

void sig_handler(int signo)
{
    std::cout << "request gateway stop\n";
    gRun = false;
}

int main(int argc, char *argv[])
{

    std::cout << "detection\n";
    signal(SIGINT, sig_handler);

    ptree configtree;
    char *iniconfig = find_char_arg(argc, argv, "-ini", "");
    std::string iniConfig(iniconfig); 

    char *xmlconfig = find_char_arg(argc, argv, "-xml", "");
    std::string xmlConfig(xmlconfig);

    char *jsonconfig = find_char_arg(argc, argv, "-json", "");
    std::string jsonConfig(jsonconfig);
    
    if (!iniConfig.empty() ? (!xmlConfig.empty() || !jsonConfig.empty()) : (!xmlConfig.empty() && !jsonConfig.empty()))
    {
        std::cout << COL_RED << "ERROR: More than one config file given.\n" << COL_END;
        return 1;
    }
	
	// ptree configtree;
    if (!iniConfig.empty())
    {
        ini_parser::read_ini(iniConfig, configtree);
    }
    else if (!xmlConfig.empty())
    {
        xml_parser::read_xml(xmlConfig, configtree);
    }
    else if (!jsonConfig.empty())
    {
        json_parser::read_json(jsonConfig, configtree);
    }

    int json_port = 0;
    std::string inputnet = std::string("yolo4_int8.rt");
    std::string inputvideo = std::string("../demo/yolo_test.mp4");
    char input_ntype = 'y';
    std::cout << "input_ntype is " << input_ntype << "\n";
    int n_classes = 80;
    int n_batch = 1;
    int show = 1;
    int save = 0;
    int ids = 0;
    int mjpeg_port = 0;
    int extyolo = 0;
    int video_mode = 0;
    int frame_rate = 30;

    if( configtree.count("tkdnn") == 0 )
    {
    // child node is missing
    }
    else
    {
        for(auto child : configtree.get_child("tkdnn"))
        {
            std::cout << COL_RED << "inside configtree.\n" << COL_END;

            if (child.first == "json_port")
                json_port = configtree.get<int>("tkdnn.json_port");
            if (child.first == "inputnet")
                inputnet = configtree.get<std::string>("tkdnn.inputnet");
            if (child.first == "inputvideo")
                inputvideo = configtree.get<std::string>("tkdnn.inputvideo");
            if (child.first == "input_ntype")
                input_ntype = configtree.get<char>("tkdnn.ntype");
            if (child.first == "n_classes")
                n_classes = configtree.get<int>("tkdnn.n_classes");
            if (child.first == "n_batch")
                n_batch = configtree.get<int>("tkdnn.n_batch");
            if (child.first == "show")
                show = configtree.get<int>("tkdnn.show");
            if (child.first == "save")
                save = configtree.get<int>("tkdnn.save");
            if (child.first == "ids")
                ids = configtree.get<int>("tkdnn.ids");
            if (child.first == "mjpeg_port")
                mjpeg_port = configtree.get<int>("tkdnn.mjpeg_port");
            if (child.first == "extyolo")
                extyolo = configtree.get<int>("tkdnn.extyolo");
            if (child.first == "video_mode")
                video_mode = configtree.get<int>("tkdnn.video_mode");
            if (child.first == "frame_rate")
                frame_rate = configtree.get<int>("tkdnn.frame_rate");
            
        // std::cout << COL_RED << "JSON_port found.\n" << COL_END;
        }
    }
    
    // JSON-Port
    json_port = find_int_arg(argc, argv, "-json_port", json_port);
    configtree.put("tkdnn.json_port", json_port);


    // Net
    char* inputnetchar = find_char_arg(argc, argv, "-net", "");
    std::string net(inputnetchar);
    if (!net.empty()) 
        inputnet = net;
    configtree.put("tkdnn.inputnet", inputnet);   

    // Input 
    char* inputvideochar = find_char_arg(argc, argv, "-input", "");
    std::string input(inputvideochar);
    if (!input.empty())
        inputvideo = input;
    configtree.put("tkdnn.inputvideo", inputvideo);   

    //Net-Type  
    char* input_ntypechar = find_char_arg(argc, argv, "-ntype", " ");
    char ntype = input_ntypechar[0];
    if (!isblank(ntype))
        input_ntype = ntype;
    configtree.put("tkdnn.ntype", input_ntype);
    std::cout << "input_ntype is " << input_ntype << "\n";

    n_classes = find_int_arg(argc, argv, "-n_classes", n_classes);
    configtree.put("tkdnn.n_classes", n_classes);
    
    n_batch = find_int_arg(argc, argv, "-n_batch", n_batch);
    configtree.put("tkdnn.n_batch", n_batch);

    show = find_int_arg(argc, argv, "-show", show);
    configtree.put("tkdnn.show", show);

    save = find_int_arg(argc, argv, "-save", save);
    SAVE_RESULT = save;
    configtree.put("tkdnn.SAVE_RESULT", SAVE_RESULT);

    ids = find_int_arg(argc, argv, "-ids", ids);
    configtree.put("tkdnn.ids", ids);

    mjpeg_port = find_int_arg(argc, argv, "-mjpeg_port", mjpeg_port);
    configtree.put("tkdnn.mjpeg_port", mjpeg_port);
  
    extyolo = find_int_arg(argc, argv, "-extyolo", extyolo);
    configtree.put("tkdnn.extyolo", extyolo);

    video_mode = find_int_arg(argc, argv, "-video_mode", video_mode);
    configtree.put("tkdnn.video_mode", video_mode);

    frame_rate = find_int_arg(argc, argv, "-frame_rate", frame_rate);
    configtree.put("tkdnn.frame_rate", frame_rate);

    if ( iniConfig.empty() && xmlConfig.empty() && jsonConfig.empty() )
    {
        std::cout << COL_GREEN << "No config file given, current configuration saved to: \"testconfiguration.ini\" \n" << COL_END;
        ini_parser::write_ini("testconfiguration.ini", configtree);
    }

    if (n_batch < 1 || n_batch > 64)
        FatalError("Batch dim not supported");


    tk::dnn::Yolo3Detection yolo;
    tk::dnn::CenternetDetection cnet;
    tk::dnn::MobilenetDetection mbnet;

    tk::dnn::DetectionNN *detNN;

    switch (input_ntype)
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

video->init(inputvideo, video_mode);

// video->

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
