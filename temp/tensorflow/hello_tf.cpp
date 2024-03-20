#include <stdio.h>
#include <tensorflow/c/c_api.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>
#include "opencv2/opencv.hpp"

class TensorFlowManager {
public:
    TensorFlowManager(const std::string& checkpointFilename = "mars-small128.pb") : checkpointFilename(checkpointFilename) {
        if (!initializeTensorFlow(checkpointFilename)) {
            // If initialization fails, throw an exception
            throw std::runtime_error("Failed to initialize TensorFlow");
        }
    }
    ~TensorFlowManager() {
        deleteSession();
    }
    void generateDetection(cv::Mat image){
        cv::Mat preprocessedImage = preprocessImage(image);
        runInference(preprocessedImage);
    }
private:
    std::string checkpointFilename;
    TF_Buffer* tf_buffer;
    TF_Graph* tf_graph;
    TF_Status* status;
    TF_SessionOptions* session_opts;
    TF_Session* session;
    TF_Operation* input_op;
    TF_Output input_var;
    TF_Operation* output_op;
    TF_Output output_var;
    int num_dims_in;
    int num_dims_out;
    int64_t* output_dims;
    int64_t* input_dims;
    bool initializeTensorFlow(const std::string checkpointFilename){
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
    void deleteSession(){
        delete[] output_dims;
        delete[] input_dims;

        TF_DeleteSession(session, status);
        TF_DeleteSessionOptions(session_opts);
        TF_DeleteGraph(tf_graph);
        TF_DeleteStatus(status);
    }
    void runInference(const cv::Mat imagePatch) {
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
                // Handle the error accordingly
            } 
            else {
                // Extract feature vectors
                std::vector<std::vector<float>> featureVectors;
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
        // Clean up input tensor
        TF_DeleteTensor(inputTensor);
    }
    cv::Mat preprocessImage(const cv::Mat& image) {
        // Resize image to match spatial dimensions (128x64)
        cv::Mat resizedImage;
        cv::resize(image, resizedImage, cv::Size(128, 64));

        // Convert image to float32 and normalize pixel values
        cv::Mat floatImage;
        resizedImage.convertTo(floatImage, CV_32FC3, 1.0 / 255.0); // Assuming pixel range [0, 255]

        // Expand dimensions to include batch dimension (-1)
        cv::Mat batchedImage;
        cv::merge(std::vector<cv::Mat>{floatImage}, batchedImage); // Add batch dimension
        
        return batchedImage;
    }
    TF_Tensor* createTensorFromMat(const cv::Mat& image) {
        // Ensure the image is not empty
        if (image.empty()) {
            return nullptr;
        }
        // Convert the image to uint8 data type
        cv::Mat uint8Image;
        image.convertTo(uint8Image, CV_8UC3); // Convert to unsigned 8-bit integers

        // Define the dimensions of the tensor
        std::vector<int64_t> dims = {1, uint8Image.cols, uint8Image.rows, uint8Image.channels()};

        // Calculate the total size of the data buffer
        size_t len = uint8Image.total() * uint8Image.elemSize();

        // Create a new tensor with the given dimensions and uint8 data type
        TF_Tensor* tensor = TF_NewTensor(TF_UINT8, dims.data(), static_cast<int>(dims.size()), uint8Image.data, len, [](void* data, size_t len, void* arg) {
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
    TF_Tensor* createOutputTensor(){
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
    void printTensorDims(TF_Tensor* tensor) {
        int numDims = TF_NumDims(tensor);
        std::cout << "Number of dimensions: " << numDims << std::endl;
        std::cout << "Dimensions: ";
        for (int i = 0; i < numDims; ++i) {
            std::cout << TF_Dim(tensor, i) << " ";
        }
        std::cout << std::endl;
    }
};

int main() {
    
    cv::Mat image = cv::imread("cow.png");
    
    if (!image.empty()) {
        cv::imshow("Image", image);
        cv::waitKey(0);  
        cv::destroyAllWindows(); 
    } else {
        std::cerr << "Failed to load image!" << std::endl;
    }

    TensorFlowManager TFM;
    
    TFM.generateDetection(image);
    
    return 0;
}
