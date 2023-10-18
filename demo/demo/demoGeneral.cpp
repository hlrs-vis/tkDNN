#include <iostream>
#include <fstream>
#include <signal.h>
#include <stdlib.h> /* srand, rand */
#include <cstdlib>
#include <unistd.h>
#include <mutex>
#include <https_stream.h> //https_stream
#include <JsonComposer.h>
#include <CSVComposer.h>

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
#include <boost/python.hpp>
#include <boost/algorithm/string.hpp>

bool gRun;
using namespace boost::property_tree;


void sig_handler(int signo)
{
    std::cout << "request gateway stop\n";
    gRun = false;
}

int main(int argc, char *argv[])
{   

    setenv("PYTHONPATH", "./deep_sort", 1);
    Py_Initialize();
    namespace python=boost::python;
    python::object my_python_class_module = python::import("MyTestClass");

    python::object ctest = my_python_class_module.attr("Test")();

    ctest.attr("awnser")("jakob");


    
    std::cout << "detection\n";
    signal(SIGINT, sig_handler);

    JsonComposer* json = NULL;
    CSVComposer* csv = NULL;

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
    
    int save_video = false;
    int black_output = false;
    int draw_detections = false;
    int inference = true;
    int csvPort = 0;
    int json_port = 0;
    std::string json_file = std::string();
    std::ofstream jsonfilestream;
    std::string csvFileName = std::string();
    std::ofstream csvFileStream;
    std::string net = std::string("yolo4_int8.rt");
    std::string inputvideo = std::string("../demo/yolo_test.mp4");
    char input_ntype = 'y';
    std::cout << "input_ntype is " << input_ntype << "\n";
    int n_classes = 80;
    int n_batch = 1;
    int show_video = 0;
    int send_video = 0;
    int ids = 0;
    int mjpeg_port = 0;
    int extyolo = 0;
    int video_mode = 0;
    int frame_rate = 30;
    int flip = 0;
    bool playback = false;
    bool save_calibration_images = 0;
    bool generate_background_image = 0;
    int calibration_frames_target = 1000;
    int calibration_frames_skip_factor = 15;
    bool continuous_background_images = true;
    std::string save_json_path = std::string("/mnt/sd-card/cameradata/json/");
    std::string save_background_image_path = std::string("/mnt/sd-card/cameradata/images/");
    std::string cfg_input = std::string("../tests/darknet/cfg/yolo4.cfg");
    std::string name_input = std::string("../tests/darknet/names/coco.names");
    std::string intrinsic_calibration_prefix = std::string("camera123");
    float conf_thresh = 0.3;
    bool adjust_exposure = 0;
    int exposure_adjust_interval = 30;
    int exposure_adjust_step = 3;
    int exposure_max_desired_mean_value = 70;
    int exposure_min_desired_mean_value = 30;
    int exposure = 50;
    int exposure_min = 3;
    int exposure_max = 2000;

    if( configtree.count("tkdnn") == 0 )
    {
    // child node is missing
    }
    else
    {
        for(auto child : configtree.get_child("tkdnn"))
        {
            // std::cout << COL_RED << "inside configtree.\n" << COL_END;

            if (child.first == "json_port")
                json_port = configtree.get<int>("tkdnn.json_port");
            if (child.first == "json_file")
                json_file = configtree.get<std::string>("tkdnn.json_file");
            if (child.first == "csvFileName")
                csvFileName = configtree.get<std::string>("tkdnn.csvFileName");
            if (child.first == "inputnet")
            {
                net = configtree.get<std::string>("tkdnn.inputnet");
                // std::cout << COL_RED << "input net: " << net << "\n" << COL_END;
            }
            if (child.first == "inputvideo")
                inputvideo = configtree.get<std::string>("tkdnn.inputvideo");
            if (child.first == "input_ntype")
                input_ntype = configtree.get<char>("tkdnn.ntype");
            if (child.first == "n_classes")
                n_classes = configtree.get<int>("tkdnn.n_classes");
            if (child.first == "n_batch")
                n_batch = configtree.get<int>("tkdnn.n_batch");
            if (child.first == "show_video")
                show_video = configtree.get<int>("tkdnn.show_video");
            if (child.first == "save_video")
                save_video = configtree.get<int>("tkdnn.save_video");
            if (child.first == "draw_detections")
                draw_detections = configtree.get<int>("tkdnn.draw_detections");
            if (child.first == "black_output")
                black_output = configtree.get<int>("tkdnn.black_output");
            if (child.first == "inference")
                inference = configtree.get<int>("tkdnn.inference");
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
            if (child.first == "flip")
                flip = configtree.get<int>("tkdnn.flip");
            if (child.first == "playback")
                playback = configtree.get<bool>("tkdnn.playback");
            if (child.first == "save_calibration_images")
                save_calibration_images = configtree.get<bool>("tkdnn.save_calibration_images");
            if (child.first == "generate_background_image")
                generate_background_image = configtree.get<bool>("tkdnn.generate_background_image");
            if (child.first == "calibration_frames_target")
                calibration_frames_target = configtree.get<int>("tkdnn.calibration_frames_target");
            if (child.first == "calibration_frames_skip_factor")
                calibration_frames_skip_factor = configtree.get<int>("tkdnn.calibration_frames_skip_factor");
            if (child.first == "continuous_background_images")
                continuous_background_images = configtree.get<bool>("tkdnn.continuous_background_images");
            if (child.first == "save_json_path")
                save_json_path = configtree.get<std::string>("tkdnn.save_json_path");
            if (child.first == "save_background_image_path")
                save_background_image_path = configtree.get<std::string>("tkdnn.save_background_image_path");
            if (child.first == "name_input")
                name_input = configtree.get<std::string>("tkdnn.name_input");
            if (child.first == "cfg_input")
                cfg_input = configtree.get<std::string>("tkdnn.cfg_input");
            if (child.first == "conf_thresh")
                conf_thresh = configtree.get<float>("tkdnn.conf_thresh");
            if (child.first == "intrinsic_calibration_prefix")
                intrinsic_calibration_prefix = configtree.get<std::string>("tkdnn.intrinsic_calibration_prefix");
            if (child.first == "adjust_exposure")
                adjust_exposure = configtree.get<bool>("tkdnn.adjust_exposure");
            if (child.first == "exposure_adjust_interval")
                exposure_adjust_interval = configtree.get<int>("tkdnn.exposure_adjust_interval");
            if (child.first == "exposure_adjust_step")
                exposure_adjust_step = configtree.get<int>("tkdnn.exposure_adjust_step");
            if (child.first == "exposure_max_desired_mean_value")
                exposure_max_desired_mean_value = configtree.get<int>("tkdnn.exposure_max_desired_mean_value");
            if (child.first == "exposure_max_desired_mean_value")
                exposure_max_desired_mean_value = configtree.get<int>("tkdnn.exposure_max_desired_mean_value");
            if (child.first == "exposure_min_desired_mean_value")
                exposure_min_desired_mean_value = configtree.get<int>("tkdnn.exposure_min_desired_mean_value");
            if (child.first == "exposure")
                exposure = configtree.get<int>("tkdnn.exposure");
            if (child.first == "exposure_min")
                exposure_min = configtree.get<int>("tkdnn.exposure_min");
            if (child.first == "exposure_max")
                exposure_max = configtree.get<int>("tkdnn.exposure_max");
        // std::cout << COL_RED << "JSON_port found.\n" << COL_END;
        }
    }

    show_video = find_int_arg(argc, argv, "-show_video", show_video);
    configtree.put("tkdnn.show_video", show_video);
    save_video = find_int_arg(argc, argv, "-save_video", save_video);
    configtree.put("tkdnn.save_video", save_video);
    send_video = find_int_arg(argc, argv, "-send_video", show_video);
    configtree.put("tkdnn.show_video", show_video);
    mjpeg_port = find_int_arg(argc, argv, "-mjpeg_port", mjpeg_port);
    configtree.put("tkdnn.mjpeg_port", mjpeg_port);
    draw_detections = find_int_arg(argc, argv, "-draw_detections", draw_detections);
    configtree.put("tkdnn.draw_detections", draw_detections);
    black_output = find_int_arg(argc, argv, "-black_output", black_output);
    configtree.put("tkdnn.black_output", black_output);

    if ( iniConfig.empty() && xmlConfig.empty() && jsonConfig.empty() )
    {
        std::cout << COL_GREEN << "No config file given, current configuration saved to: \"testconfiguration.ini\" \n" << COL_END;
        ini_parser::write_ini("testconfiguration.ini", configtree);
    }
    else
    {
        std::cout << COL_GREEN << "config file given, current configuration saved to: \"testconfiguration2.ini\" \n" << COL_END;
        ini_parser::write_ini("testconfiguration2.ini", configtree);
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

    //    detNN->init(net, n_classes, n_batch);
    detNN->init(net,cfg_input,name_input,n_classes,n_batch,conf_thresh);


    gRun = true;

    if (mjpeg_port > 0)
    {
	    send_video = true;
    }

bool video_output = (show_video || save_video || send_video);

bool write_json;
bool writeCsv;

std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
std::time_t t_c = std::chrono::system_clock::to_time_t(now);
char date[100];
std::strftime(date, sizeof(date), "_%F_%H-%M-%S", std::localtime(&t_c));
auto hours = std::chrono::duration_cast<std::chrono::hours>(now.time_since_epoch()).count();
int day_recording_started = int(hours/24);

//std::cout << "Hours since epoch: " << int(hours) << std::endl;
//std::cout << "Days since epoch: " << int(hours/24) << std::endl;
//auto tomorrow = now + std::chrono::hours(24);
//auto tomorrow_hours = std::chrono::duration_cast<std::chrono::hours>(tomorrow.time_since_epoch()).count();
//std::cout << "Tomorrow Hours since epoch: " << int(tomorrow_hours) << std::endl;
//std::cout << "Tomorrow Days since epoch: " << int(tomorrow_hours/24) << std::endl;

// std::cout << COL_RED << "json_file size" << json_file.size() << "\n" << COL_END;
if (json_file.size()>0){    
    write_json = true;
    std::string json_extension = ".json";
    if (boost::algorithm::iends_with(json_file,json_extension))
        json_file = json_file.substr(0, json_file.size()-5);
    json_file = json_file.append(date);
    json_file = json_file.append(".json");
    jsonfilestream.open(json_file);
    jsonfilestream << "[";
}
std::cout << csvFileName << "\n";
if (csvFileName.size() > 0){
    
    writeCsv = true;
    
    csvFileStream.open(csvFileName);
    
    csv = new CSVComposer;
}
else writeCsv = false;
if (writeCsv == true){
csv->initiate(csvFileName, csvFileStream, inputvideo);
}

if (write_json || json_port > 0){
    json = new JsonComposer;
}

VideoAcquisition *video;

if (!ids){
    video = new OpenCVVideoCapture;
}
else{
    video = new IDSVideoCapture;
}

video->init(inputvideo, video_mode);

if (playback){
    // Slow down image acquisition instead of removing images from queue
    video->setPlayback();
}

if (flip){
    video->flip();
}
if (adjust_exposure){
    video->setAdjustExposure();
    video->set_exposure_adjust_interval(exposure_adjust_interval);
    video->set_exposure_adjust_step(exposure_adjust_step);
    video->set_exposure_max_desired_mean_value(exposure_max_desired_mean_value);
    video->set_exposure_min_desired_mean_value(exposure_min_desired_mean_value);
    video->set_exposure_min(exposure_min);
    video->set_exposure_max(exposure_max);
    video->setExposure(exposure);
}


cv::Size image_size= cv::Size(video->getWidth(), video->getHeight());

cv::Mat avgImgConverted(image_size, CV_64FC3);
Mat H(image_size, CV_64FC3, 0.0);

video->start();

if (json){
    json->setResolution(video->getWidth(), video->getHeight());
}

    cv::VideoWriter resultVideo;
    if (save_video){   
        int w = video->getWidth();
        int h = video->getHeight();
	std::string resultVideoFile = "Videoresult";
	resultVideoFile = resultVideoFile.append(date);
	resultVideoFile = resultVideoFile.append(".mp4");
        resultVideo.open(resultVideoFile, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, cv::Size(w, h));

        std::cout << "Result video initialized, saving as "  << resultVideoFile << std::endl;
    }

    cv::Mat frame;
    if (show_video){
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
    int json_frames_written = 0;
    int calibration_frames_taken = 0;


    while (gRun){
        //ensure queue holds enough pictures for batch size

        batch_dnn_input.clear();
        batch_frame.clear();
        batch_images->clear();

        gRun = video->getImages(batch_images, n_batch);
        
        for (int bi = 0; bi < n_batch; ++bi){
            // this will be used for the visualisation
            if (video_output || save_calibration_images)
                batch_frame.push_back((*batch_images)[bi].data);

            if (generate_background_image){
                if (calibration_frames_taken < calibration_frames_target){
                    if (frames_processed % calibration_frames_skip_factor == 0){
                        cv::Mat accFrame = (*batch_images)[bi].data;

                        for(int i = 0; i < H.rows; i++)
                            for(int j = 0; j < H.cols; j++){
                                H.at<Vec3d>(i,j)[0]+=double(accFrame.at<Vec3b>(i,j)[0]);
                                H.at<Vec3d>(i,j)[1]+=double(accFrame.at<Vec3b>(i,j)[1]);
                                H.at<Vec3d>(i,j)[2]+=double(accFrame.at<Vec3b>(i,j)[2]);
                            }

                        calibration_frames_taken++;
                        avgImgConverted = H;
                        avgImgConverted.convertTo(avgImgConverted, CV_8UC3, 1.0/calibration_frames_taken);

                        if (show_video)
                            cv::imshow("averageConverted", avgImgConverted);
                    }
                }
                else if (calibration_frames_taken == calibration_frames_target){
                    cv::String outFileName = save_background_image_path;
                    outFileName.append("/background_image");
                    //cv::String outFileName = "background_image";
                    outFileName.append(date);
                    now = std::chrono::system_clock::now();
                    t_c = std::chrono::system_clock::to_time_t(now);
                    //check for date change, if a new day has started end program to be restarted
                    hours = std::chrono::duration_cast<std::chrono::hours>(now.time_since_epoch()).count();
                    int day_now = int(hours/24);
                    if (day_now != day_recording_started)
                        gRun = false;
                    std::strftime(date, sizeof(date), "_%F_%H-%M-%S", std::localtime(&t_c));
                    outFileName.append(".jpg");
                    std::cout << COL_RED << "Trying to save" << outFileName << "\n" << COL_END;
		            cv::imwrite(outFileName,avgImgConverted);
                    std::cout << "background image saved after " << frames_processed << "processed frames, " << calibration_frames_taken << " frames used." << std::endl;
                    if (continuous_background_images){
                        calibration_frames_taken = 0;
                        //H.zeros(image_size,CV_64FC3);
                        for(int i = 0; i < H.rows; i++)
                            for(int j = 0; j < H.cols; j++){
                                H.at<Vec3d>(i,j)[0]=0.0;
                                H.at<Vec3d>(i,j)[1]=0.0;
                                H.at<Vec3d>(i,j)[2]=0.0;
                            }
                    }
                    else{
                        generate_background_image = false;
                    }
                }
            }
            else{
                if (frames_processed % 1000 == 0){
                    now = std::chrono::system_clock::now();
                    t_c = std::chrono::system_clock::to_time_t(now);
                    //check for date change, if a new day has started end program to be restarted
                    hours = std::chrono::duration_cast<std::chrono::hours>(now.time_since_epoch()).count();
                    int day_now = int(hours/24);
                if (day_now != day_recording_started)
                    gRun = false;
                }
            }
            // this will be resized to the net format
            if (!video_output)
                //batch_dnn_input.push_back((*batch_images)[bi].data);
                batch_dnn_input.push_back(std::move((*batch_images)[bi].data));
            else
                batch_dnn_input.push_back((*batch_images)[bi].data);
            frames_processed += 1;
        }

        if (save_calibration_images){
            if (frames_processed % 100 == 0){
                cv::String outFileName = intrinsic_calibration_prefix + std::to_string(frames_processed);
                outFileName.append(".jpg");
                cv::imwrite(outFileName,batch_frame[0]);
                std::cout << "saved:" << outFileName << std::endl;
            }
        }

        //inference
        if (inference)
            detNN->update(batch_dnn_input, n_batch);


	    // Video block, images are used for inference/background calculation and will be manipulated for output/visualizationpurposes
        if (video_output){
            if (black_output){
	            for (int bi = 0; bi < n_batch; ++bi){
                    batch_frame[bi].setTo(cv::Scalar::all(0));
                }
            }
            if (draw_detections){
                if (inference)
                    detNN->draw(batch_frame,extyolo);
            }
            if (show_video){
                for (int bi = 0; bi < n_batch; ++bi)
                {
                    cv::imshow("detection", batch_frame[bi]);
                }
                //cv::imshow("detection", batch_frame[0]);
            }
            if (cv::waitKey(1) == 27){
                break;
            }
            if (save_video){
                for (int bi = 0; bi < n_batch; ++bi)
                {
                    resultVideo << batch_frame[bi];
                }
                //resultVideo << batch_frame[0];
            }   
            if (mjpeg_port > 0){
                send_mjpeg(batch_frame[0], mjpeg_port, 400000, 40);
            }
	    }
        if (write_json || json_port > 0){
            //send_json(batch_images, *detNN, json_port, 40000);
            char *send_buf = json->detection_to_json(batch_images, *detNN, NULL);
            if (json_port > 0)
            {
                send_json(send_buf, json_port, 40000);
            }
            if (!json_file.empty())
            {
                if (json_frames_written != 0)
                {
                    jsonfilestream << ",\n";
                }
                jsonfilestream << send_buf;
                json_frames_written++;
            }
            free(send_buf);
        }
        if (writeCsv > 0) {
            csv->detectionToCsv(batch_images, *detNN, csvFileStream);
        }
    }
    video->stop();
    jsonfilestream << "]";
    jsonfilestream.close();
    csvFileStream.close();
    resultVideo.release();
    long long int frame_id = (*batch_images)[n_batch-1].frame_id;

    std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();

    std::cout << "detection end\n";
    double mean = 0;
    if (inference){
        std::cout << COL_GREENB << "\n\nTime stats:\n";
        std::cout << "Min: " << *std::min_element(detNN->stats.begin(), detNN->stats.end()) / n_batch << " ms\n";
        std::cout << "Max: " << *std::max_element(detNN->stats.begin(), detNN->stats.end()) / n_batch << " ms\n";
        for (int i = 0; i < detNN->stats.size(); i++)
            mean += detNN->stats[i];
        mean /= detNN->stats.size();
        std::cout << "Avg: " << mean / n_batch << " ms\t" << 1000 / (mean / n_batch) << " FPS\n"
                << COL_END;
    }
    std::cout << COL_GREENB << "Frames overall: " << frames_processed / std::chrono::duration<double>(end_time-start_time).count() << " fps \n" << COL_END;

    return 0;
}
