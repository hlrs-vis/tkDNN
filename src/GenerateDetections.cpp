#include "GenerateDetections.h"
#include "DetectionWithFeatureVector.h"

ImageEncoder::ImageEncoder() : tfm() {

}

cv::Mat ImageEncoder::extract_image_patch(const cv::Mat &image, const cv::Rect &bbox, const cv::Size &patch_shape) {
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

cv::Mat ImageEncoder::preprocessPatch(const cv::Mat& image) {
    // Resize image to match spatial dimensions (128x64)
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(128, 64));
    // Convert the image to uint8 data type
    cv::Mat uint8Image;
    resizedImage.convertTo(uint8Image, CV_8UC3, 1.0/ 255.0); // Convert to unsigned 8-bit integers

    // Expand dimensions to include batch dimension (-1)
    cv::Mat batchedImage;
    cv::merge(std::vector<cv::Mat>{uint8Image}, batchedImage); // Add batch dimension
    
    return batchedImage;
}

std::vector<std::vector<DetectionWithFeatureVector>> ImageEncoder::generateDetections(std::vector<TypewithMetadata<cv::Mat>> *batch_images, tk::dnn::DetectionNN &detNN) { 
    
    std::vector<std::vector<DetectionWithFeatureVector>> detections_out;
    for (int bi = 0; bi < detNN.batchDetected.size(); ++bi){        // Iterate the frames in one batch
        vector<cv::Rect> boxes;
        for (int i = 0; i < detNN.batchDetected[bi].size(); i++){   // Iterate the detections in one frame
            tk::dnn::box b;
            b = detNN.batchDetected[bi][i];  
            cv::Rect box(static_cast<int>(b.x), static_cast<int>(b.y), static_cast<int>(b.w), static_cast<int>(b.h));             
            boxes.push_back(box); // Safe the x,y-coordinates, width and height
            cv::Mat patch = extract_image_patch((*batch_images)[bi].data, box, cv::Size(static_cast<int>(b.w), static_cast<int>(b.h)));
            cv::Mat preprossedPatch = preprocessPatch(patch);
            std::vector<float> featureVector = tfm.generateFeatureVector(preprossedPatch);
            detections_out[bi].push_back({(*batch_images)[bi].frame_id, b.x, b.y, b.w, b.h, b.prob, featureVector}); // Safe the detections in the MOT challenge format

        }
        
    }

    return detections_out;
}


TensorFlowManager::TensorFlowManager(const std::string& checkpointFilename = "mars-small128.pb") : checkpointFilename(checkpointFilename) {
    if (!initializeTensorFlow(checkpointFilename)) {
        // If initialization fails, throw an exception
        throw std::runtime_error("Failed to initialize TensorFlow");
    }
}
TensorFlowManager::~TensorFlowManager() {
    deleteSession();
}
std::vector<float> TensorFlowManager::generateFeatureVector(cv::Mat image){
    std::vector<float> featureVector;
    featureVector = runInference(image);
    return featureVector;
}
bool TensorFlowManager::initializeTensorFlow(const std::string checkpointFilename){
    printf("Hello you from TensorFlow C library version %s\n", TF_Version());

    // Read in data from file
    std::ifstream file(checkpointFilename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error opening tf_graph file: " << checkpointFilename << std::endl;
        return false;
    }
    else {
        fprintf(stdout, "Successfully opened tf_graph file \n");
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(file_size);
    if (!file.read(buffer.data(), file_size)) {
        std::cerr << "Error reading tf_graph file: " << checkpointFilename << std::endl;
        return false;
    }
    else {
        fprintf(stdout, "Successfully read tf_graph file\n");
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
        TF_DeleteStatus(status);
        TF_DeleteGraph(tf_graph);
        return false;
    }
    else { 
        fprintf(stdout, "Successfully imported graph: %s\n", TF_Message(status));
    }
    // Start a new TF_Session
    session_opts = TF_NewSessionOptions();
    session = TF_NewSession(tf_graph, session_opts, status);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error creating session: %s\n", TF_Message(status));
        TF_DeleteStatus(status);
        TF_DeleteGraph(tf_graph);
        return false;
    }
    else {
        fprintf(stdout, "Successfully created session: %s\n", TF_Message(status));
    }
    // Get the input and output graph 
    input_op = TF_GraphOperationByName(tf_graph, "images");
    if (input_op == nullptr) {
        fprintf(stderr, "Error: Failed to find operation 'images' in the graph.\n");
        TF_DeleteStatus(status);
        TF_DeleteGraph(tf_graph);
        return false;
    }
    input_var = {input_op, 0};
    output_op = TF_GraphOperationByName(tf_graph, "features");
    if (output_op == nullptr) {
        fprintf(stderr, "Error: Failed to find operation 'features' in the graph.\n");
        TF_DeleteStatus(status);
        TF_DeleteGraph(tf_graph);
        return false;
    }
    output_var = {output_op, 0};
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error getting graph by name: %s\n", TF_Message(status));
        TF_DeleteStatus(status);
        TF_DeleteGraph(tf_graph);
        return false;
    }

    // Get the output tensor shape and feature dimensions
    output_dims = nullptr;
    int num_dims_out = TF_GraphGetTensorNumDims(tf_graph, output_var, status);
    if (num_dims_out < 0) {
        fprintf(stderr, "Error getting tensor dimensions for output_var\n");
        // Clean up and return error code
        TF_DeleteStatus(status);
        TF_DeleteSession(session, status);
        TF_DeleteGraph(tf_graph);
        return false;
    }
    else if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Output is not in graph %s\n", TF_Message(status));
        TF_DeleteStatus(status);
        TF_DeleteGraph(tf_graph);
        return false;
    }
    else {
        fprintf(stdout, "Successfully got tensor dimensions %s\n", TF_Message(status));
    }

    output_dims = new int64_t[num_dims_out];
    if (output_dims == nullptr) {
        fprintf(stderr, "Error allocating memory for output_dims\n");
        // Clean up and return error code
        TF_DeleteStatus(status);
        TF_DeleteSession(session, status);
        TF_DeleteGraph(tf_graph);
        return false;
    }
    TF_GraphGetTensorShape(tf_graph, output_var, output_dims, num_dims_out, status);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error getting tensor shape for output_var: %s\n", TF_Message(status));
        // Clean up and return error code
        delete[] output_dims;
        TF_DeleteStatus(status);
        TF_DeleteSession(session, status);
        TF_DeleteGraph(tf_graph);
        return false;
    }
    int feature_dim = output_dims[num_dims_out -1];

    // Get the input tensor shape
    input_dims = nullptr;
    int num_dims_in = TF_GraphGetTensorNumDims(tf_graph, input_var, status);
    if (num_dims_in < 0) {
        fprintf(stderr, "Error getting tensor dimensions for input_var\n");
        // Clean up and return error code
        delete[] output_dims;
        TF_DeleteStatus(status);
        TF_DeleteSession(session, status);
        TF_DeleteGraph(tf_graph);
        return false;
    }
    input_dims = new int64_t[num_dims_in];
    if (input_dims == nullptr) {
        fprintf(stderr, "Error allocating memory for input_dims\n");
        // Clean up and return error code
        delete[] output_dims;
        TF_DeleteStatus(status);
        TF_DeleteSession(session, status);
        TF_DeleteGraph(tf_graph);
        return false;
    }
    TF_GraphGetTensorShape(tf_graph, input_var, input_dims, num_dims_in, status);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error getting tensor shape for input_var: %s\n", TF_Message(status));
        // Clean up and return error code
        delete[] output_dims;
        delete[] input_dims;
        TF_DeleteStatus(status);
        TF_DeleteSession(session, status);
        TF_DeleteGraph(tf_graph);
        return false;
    }
    int image_shape = input_dims[3];
    // Check the rank of the input and output graph
    assert(num_dims_in == 4); // Check the number of dimensions
    std::cout << "Image Shape: " << input_dims[1] << ";" << input_dims[2] << ";" << input_dims[3] << std::endl;
    assert(num_dims_out == 2); // Check the number of dimensions
    std::cout << "Feature Dimension: " << feature_dim << std::endl;
    TF_DataType outputDataType = TF_OperationOutputType({output_op, 0}); // Assuming index 0
    std::cout << "Tensor Data Type (Integer Representation): " << outputDataType << std::endl;

    return true;
}
void TensorFlowManager::deleteSession(){
    delete[] output_dims;
    delete[] input_dims;

    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(session_opts);
    TF_DeleteGraph(tf_graph);
    TF_DeleteStatus(status);
}
std::vector<float> TensorFlowManager::runInference(const cv::Mat imagePatch) {
    std::cout << "Test" << std::endl;
    TF_Tensor* inputTensor = createTensorFromMat(imagePatch);
    TF_Tensor* inputTensorPtr = inputTensor;
    std::cout << "Test" << std::endl;
    TF_Tensor* outputTensor = createOutputTensor();
    TF_Tensor* outputTensorPtr = outputTensor;
    // Run inference
    TF_SessionRun(session,          // TF_Session*    session
                    nullptr,          // TF_Buffer,     run options (nullptr for default options)
                    &input_var,       // TF_Output*,    input Tensors
                    &inputTensorPtr,  // TF_Tensor*,    input Values
                    1,                // int,           number of input tensors
                    &output_var,      // TF_Output,     output tensors
                    &outputTensorPtr, // TF_Tensor,     output tensor values
                    1,                // int,           number of output tensors
                    nullptr,          // TF_Operation*, output operations
                    0,                // int,           number of targets
                    nullptr,          // TF_Buffer*,    run metadata (nullptr for default options)
                    status            // TF_Status*,    status
    );
    std::cout << "Test" << std::endl;
    std::vector<std::vector<float>> featureVectors;
    if (TF_GetCode(status) == TF_OK && outputTensor != nullptr) {
        // Access the data pointer of the output tensor
        float* outputData = static_cast<float*>(TF_TensorData(outputTensor));
        
        // Get the shape of the output tensor
        std::vector<int64_t> outputShape(2);
        int numDims = TF_NumDims(outputTensor);
        for (int i = 0; i < numDims; ++i) {
            outputShape[i] = TF_Dim(outputTensor, i);
        }
        // Ensure outputData is not nullptr
        if (outputData == nullptr) {
            std::cerr << "Error: Output tensor data is null." << std::endl;               
        } 
        else {
            // Extract feature vectors
            int batchSize = outputShape[0];
            int featureSize = outputShape[1];

            for (int i = 0; i < batchSize; ++i) {
                std::vector<float> featureVector(outputData + i * featureSize, outputData + (i + 1) * featureSize);
                featureVectors.push_back(featureVector);
            }
            // Print the received data
            std::cout << "Received data:" << std::endl;
            for (int i = 0; i < outputShape[0]; ++i) {
                std::cout << "Batch " << i << ":" << std::endl;
                for (int j = 0; j < outputShape[1]; ++j) {
                    std::cout << outputData[i * outputShape[1] + j] << " ";
                }
                std::cout << std::endl;
            }
        }

        // Clean up output tensor
        TF_DeleteTensor(outputTensor);
    } 
    else {
        // Handle error
        std::cerr << "Error performing inference: " << TF_Message(status) << std::endl;
    }
    TF_DeleteTensor(inputTensor);

    return featureVectors[0];
}
TF_Tensor* TensorFlowManager::createTensorFromMat(const cv::Mat& image) {
    // Ensure the image is not empty
    if (image.empty()) {
        return nullptr;
    }

    // Define the dimensions of the tensor
    std::vector<int64_t> dims = {1, image.cols, image.rows, image.channels()};

    // Calculate the total size of the data buffer
    size_t len = image.total() * image.elemSize();

    // Create a new tensor with the given dimensions and uint8 data type
    TF_Tensor* tensor = TF_NewTensor(TF_UINT8, dims.data(), static_cast<int>(dims.size()), image.data, len, [](void* data, size_t len, void* arg) {
        // No need to deallocate memory here since we're using the original image data
    }, nullptr);
    if (tensor == nullptr) {
        // Error handling: Log an error message or take appropriate action
        std::cerr << "Error creating outputTensor." << std::endl;
    } else {
        // Output tensor created successfully
        printTensorDims(tensor);
    }

    return tensor;
}
TF_Tensor* TensorFlowManager::createOutputTensor(){
    int64_t outputDims[] = {1, 128}; // Assuming the output shape is (1, 128)
    int numDimsOut = 2; // Number of dimensions for the output tensor
    // Calculate the total size of the tensor data
    size_t outputDataSize = 1;
    for (int i = 0; i < numDimsOut; ++i) {
        outputDataSize *= outputDims[i];
    }
    // Allocate memory for the output tensor data and initialize to zeros
    float* outputData = new float[outputDataSize](); // Zero initialization
    // Create the output tensor
    TF_Tensor* outputTensor = TF_NewTensor(TF_FLOAT, outputDims, numDimsOut, outputData, outputDataSize * sizeof(float), [](void* data, size_t, void*) {
        delete[] static_cast<float*>(data); // Deallocate memory
    }, nullptr);
    if (outputTensor == nullptr) {
        // Error handling: Log an error message or take appropriate action
        std::cerr << "Error creating outputTensor." << std::endl;
        // Additional error handling logic here...
    } else {
        // Output tensor created successfully
        printTensorDims(outputTensor);
    }
    return outputTensor;
}
void TensorFlowManager::printTensorDims(TF_Tensor* tensor) {
    int numDims = TF_NumDims(tensor);
    std::cout << "Number of dimensions: " << numDims << std::endl;
    std::cout << "Dimensions: ";
    for (int i = 0; i < numDims; ++i) {
        std::cout << TF_Dim(tensor, i) << " ";
    }
    std::cout << std::endl;
}



