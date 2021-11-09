#ifndef JSON_COMPOSER_H
#define JSON_COMPOSER_H

#include "TypewithMetadata.h"
#include "DetectionNN.h"

using namespace cv;
using namespace tk; 


// JSON format:
//{
// "frame_id":8990, "frame_time":1631555332.334568
// "objects":[
//  {"class_id":4, "name":"aeroplane", "relative coordinates":{"center_x":0.398831, "center_y":0.630203, "width":0.057455, "height":0.020396}, "confidence":0.793070},
//  {"class_id":14, "name":"bird", "relative coordinates":{"center_x":0.398831, "center_y":0.630203, "width":0.057455, "height":0.020396}, "confidence":0.265497}
// ]
//},

char *detection_to_json(std::vector<TypewithMetadata<cv::Mat>> *batch_images, tk::dnn::DetectionNN &detNN, char *filename)
{
    float Yx, Yy, Yw, Yh;
    cv::Size sz = (*batch_images)[0].data.size();
    int imageWidth = sz.width;
    int imageHeight = sz.height;
    std::string det_class;
    char *send_buf = (char *)calloc(2048, sizeof(char));
    if (!send_buf) return 0;
    
    tk::dnn::box b;

    sprintf(send_buf, "");
    int batch_index =-1;

    for (int bi = 0; bi < detNN.batchDetected.size(); ++bi)
    {
        if (batch_index != -1) strcat(send_buf, ", \n");
        batch_index=bi;
        char *header_buf = (char *)calloc(2048, sizeof(char));
        if (filename) {
            sprintf(header_buf, "{\n \"frame_id\":%lld, \n \"filename\":\"%s\", \n \"objects\": [ \n", (*batch_images)[bi].frame_id, filename);
        }
        else {
            if (true){
                //auto time1 = *frame_times;
                //auto time2 = time1[bi];
                auto time2 = (*batch_images)[bi].time;
                auto time3 = std::chrono::duration<double>(time2.time_since_epoch()).count();
                sprintf(header_buf, "{\n \"frame_id\":%lld,\"frame_time\":%f, \n \"objects\": [ \n", (*batch_images)[bi].frame_id,time3);
            }
            else
            sprintf(header_buf, "{\n \"frame_id\":%lld, \n \"objects\": [ \n", (*batch_images)[bi].frame_id);
        }
        int send_buf_len = strlen(send_buf);
        int header_buf_len = strlen(header_buf);
        int total_len = send_buf_len + header_buf_len + 100;
        send_buf = (char *)realloc(send_buf, total_len * sizeof(char));
        strcat(send_buf, header_buf);
        free(header_buf);
        // draw dets
        int json_index = -1;
        for (int i = 0; i < detNN.batchDetected[bi].size(); i++)
        {
            if (json_index != -1) strcat(send_buf, ", \n");
            json_index = i;
            b = detNN.batchDetected[bi][i];
            det_class = detNN.classesNames[b.cl];


            //auto time3 = std::chrono::duration_cast<double, std::chrono::milliseconds>(time2.time_since_epoch()).count();

            //yolo stuff

            Yx = (b.x + (int)(b.w / 2)) / imageWidth;
            Yy = (b.y + (int)(b.h / 2)) / imageHeight;
            Yw = b.w / imageWidth;
            Yh = b.h / imageHeight;

            char *buf = (char *)calloc(2048, sizeof(char));
            sprintf(buf, "  {\"class_id\":%d, \"name\":\"%s\", \"relative_coordinates\":{\"center_x\":%f, \"center_y\":%f, \"width\":%f, \"height\":%f}, \"confidence\":%f}",
            b.cl, det_class.c_str(), Yx, Yy, Yw, Yh, b.prob);
            send_buf_len = strlen(send_buf);
            int buf_len = strlen(buf);
            total_len = send_buf_len + buf_len + 100;
            send_buf = (char *)realloc(send_buf, total_len * sizeof(char));
            strcat(send_buf, buf);
            free(buf);
        }
        strcat(send_buf, "\n ] \n} \n");
    }

    return send_buf;
}


#endif /*JSON_COMPOSER_H*/