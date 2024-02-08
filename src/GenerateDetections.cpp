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

    // Read in data from file
    std::ifstream file(checkpoint_filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error opening tf_graph file: " << checkpoint_filename << std::endl;
        return 1;
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(file_size);
    if (!file.read(buffer.data(), file_size)) {
        std::cerr << "Error reading tf_graph file: " << checkpoint_filename << std::endl;
        return 1;
    }

    // Create empty buffer and fill it with read in data
    tf_buffer = TF_NewBuffer();
    tf_buffer->data = buffer.data();
    tf_buffer->length = buffer.size();
   
    // Create empty graph
    tf_graph = TF_NewGraph();
    status = TF_NewStatus();
    
    //File Graph with data from buffer
    TF_ImportGraphDefOptions* tf_graph_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(tf_graph, tf_buffer, tf_graph_opts, status);
    TF_DeleteImportGraphDefOptions(tf_graph_opts);
    TF_DeleteBuffer(tf_buffer);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error importing tf_graph: %s\n", TF_Message(status));
        return 1;
    }

    // Start a new TF_Session
    session_opts = TF_NewSessionOptions();
    session = TF_NewSession(tf_graph, session_opts, status);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error creating session: %s\n", TF_Message(status));
        return 1;
    }

    // Get the input and output graph 
    input_var = {TF_GraphOperationByName(tf_graph, "net/input_name"), 0};
    output_var = {TF_GraphOperationByName(tf_graph, "net/output_name"), 0};
    
    // Check the rank of the input and output graph
    assert(TF_OperationOutputType(input_var) == TF_FLOAT);
    assert(TF_NumDims(TF_GraphGetTensorShape(tf_graph, input_var, status)) == 4);
    assert(TF_NumDims(TF_GraphGetTensorShape(tf_graph, output_var, status)) == 2);
    
    // Get the output tensor shape and feature dimensions
    num_dims_out = TF_GraphGetTensorNumDims(tf_graph, output_var, status);
    TF_GraphGetTensorShape(tf_graph, output_var, dims, num_dims_out, status);
    int feature_dim = dims[num_dims_out - 1];
    
    // Get the input tensor shape
    num_dims_in = TF_GraphGetTensorNumDims(tf_graph, input_var, status);
    TF_GraphGetTensorShape(tf_graph, input_var, image_shape, num_dims_in, status);
    
}

void ImageEncoder::deletingSession() {
    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(session_opts);
    TF_DeleteGraph(tf_graph);
    TF_DeleteStatus(status);
}

std::vector<cv::Mat> ImageEncoder::call(std::vector<cv::Mat> &data) {
    //size_t data_lenght = data.size();
    //vector<vector<float>> out(data_lenght , std::vector<float>(feature_dim, 0.0f));
    
}

std::function<std::vector<cv::Mat>(const cv::Mat&, const std::vector<cv::Rect>&)> createBoxEncoder(const std::string &model_filename, const std::string &input_name = "images", const std::string &output_name = "features") {
    ImageEncoder image_encoder;
    image_encoder.initialization(model_filename, input_name, output_name);
    cv::Size image_shape(image_encoder.image_shape[0],image_encoder.image_shape[1]);

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

std::vector<DetectionWithFeatureVector> generateDetections(std::function<auto(cv::Mat, vector<cv::Rect>)> encoder, std::vector<TypewithMetadata<cv::Mat>> *batch_images, tk::dnn::DetectionNN &detNN) { 
    
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


