#include "GenerateDetections.h"
#include "DetectionWithFeatureVector.h"

cv::Mat extract_image_patch(const cv::Mat &image, const cv::Rect &bbox, const cv::Size &patch_shape) {
    cv::Rect patched_bbox = bbox;

    if (patch_shape.width > 0 && patch_shape.height > 0) {
        double target_aspect = static_cast<double>(patch_shape.width) / patch_shape.height;
        double new_width = target_aspect * patched_bbox.height;
        patched_bbox.x -= (new_width - patched_bbox.width) / 2;
        patched_bbox.width = new_width;
    }

    // Convert to top left, bottom right
    patched_bbox.width += patched_bbox.x;
    patched_bbox.height += patched_bbox.y;

    // Clip at image boundaries
    patched_bbox.x = std::max(0, patched_bbox.x);
    patched_bbox.y = std::max(0, patched_bbox.y);
    patched_bbox.width = std::min(image.cols - 1, patched_bbox.width);
    patched_bbox.height = std::min(image.rows - 1, patched_bbox.height);

    if (patched_bbox.x >= patched_bbox.width || patched_bbox.y >= patched_bbox.height) {
        return cv::Mat();
    }

    cv::Mat image_patch = image(patched_bbox);
    cv::resize(image_patch, image_patch, patch_shape);
    return image_patch;
}

ImageEncoder::ImageEncoder() {

}

int ImageEncoder::initialization(const std::string &checkpoint_filename, const std::string &input_name = "images", const std::string &output_name = "features") {
    TF_Graph* graph = TF_NewGraph();
    TF_Status* status = TF_NewStatus();

    TF_SessionOptions* session_opts = TF_NewSessionOptions();
    TF_Buffer* buffer = TF_ReadFile("checkpoint_filename", status);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error reading file: %s\n", TF_Message(status));
        return 1;
    }

    TF_ImportGraphDefOptions* graph_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, buffer, graph_opts, status);
    TF_DeleteImportGraphDefOptions(graph_opts);
    TF_DeleteBuffer(buffer);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error importing graph: %s\n", TF_Message(status));
        return 1;
    }

    TF_Session* session = TF_NewSession(graph, session_opts, status);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error creating session: %s\n", TF_Message(status));
        return 1;
    }

    TF_Output input_op = {TF_GraphOperationByName(graph, "net/input_name"), 0};
    TF_Output output_op = {TF_GraphOperationByName(graph, "net/output_name"), 0};

    assert(TF_OperationOutputType(input_op) == TF_FLOAT);
    assert(TF_NumDims(TF_GraphGetTensorShape(graph, input_op, status)) == 4);
    assert(TF_NumDims(TF_GraphGetTensorShape(graph, output_op, status)) == 2);

    // Get the output tensor shape and feature dimensions
    int64_t* output_shape = TF_GraphGetTensorShape(graph, output_op, status)->dims;
    int num_dims = TF_GraphGetTensorShape(graph, output_op, status)->num_dims;
    int feature_dim = output_shape[num_dims - 1];

    // Get the input tensor shape
    int64_t* input_shape = TF_GraphGetTensorShape(graph, input_op, status)->dims;
    
}

void deletingSession() {
    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(session_opts);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
}

vector<vector<float>> ImageEncoder::call(const cv::Mat &data) {
    size_t data_lenght = data.size();
    vector<vector<float>> out(data_lenght , std::vector<float>(feature_dim, 0.0f));
    
}

cv::Mat createBoxEncoder(const std::string &model_filename, const std::string &input_name = "images", const std::string &output_name = "features") {
    ImageEncoder image_encoder;
    image_encoder.initialization(model_filename, input_name, output_name);
    cv::Size image_shape = image_encoder.image_shape;

    auto encoder = [&](const cv::Mat& image, const vector<cv::Rect>& boxes) {
        std::vector<cv::Mat> image_patches;
        for (const cv::Rect& box : boxes) {
            cv::Mat patch = extract_image_patch(image, box, cv::Size(image_shape.width, image_shape.height));
            if (patch.empty()) {
                std::cout << "WARNING: Failed to extract image patch: " << box << "." << std::endl;
                patch = cv::Mat(image_shape, CV_8UC3); 
                cv::randu(patch, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
            }
            image_patches.push_back(patch);
        }

        return image_encoder.call(image_patches);
    };

    return encoder;
}

std::vector<DetectionWithFeatureVector> generateDetections(std::function<auto(cv::Mat, vector<cv::Rect>)> encoder, std::vector<TypewithMetadata<cv::Mat>> *batch_images, tk::dnn::detectionNN &detNN) { 
    
    std::vector<DetectionWithFeatureVector> detections_out;
    for (int bi = 0; bi < detNN.batchDetected.size(); ++bi){        // Iterate the frames in one batch
        vector<cv::Rect> boxes;
        for (int i = 0; i < detNN.batchDetected[bi].size(); i++){   // Iterate the detections in one frame
            tk::dnn::box b;
            float gX = -1, gY = -1, gZ = -1;
            b = detNN.batchDetected[bi][i];                
            boxes.push_back(cv::Rect(static_cast<int>(b.x), static_cast<int>(b.y), static_cast<int>(b.w), static_cast<int>(b.h))); // Safe the x,y-coordinates, width and height
            detections_out.push_back({(*batch_images)[bi].frame_id, b.cl, b.x, b.y, b.w, b.h, b.prob, gX, gY, gZ}); // Safe the detections in the MOT challenge format
        }
        vector<vector<float>> features = encoder(batch_images[bi].data, boxes); // Call encoder with all detections for one frame
        
        // Fill the feature vector
        for (int i = 0; i < features.size(); i++) {     
            detections_out[i].feature_vector = (features[i]);
        }
        boxes.clear();
    }

    return detections_out;
}


