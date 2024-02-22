#include <stdio.h>
#include <tensorflow/c/c_api.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    printf("Hello from TensorFlow C library version %s\n", TF_Version());

    // Read in data from file
    std::string checkpoint_filename = "mars-small128.pb";
    std::ifstream file(checkpoint_filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error opening tf_graph file: " << checkpoint_filename << std::endl;
        return 1;
    }
    else {
        fprintf(stdout, "Successfully opened tf_graph file \n");
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(file_size);
    if (!file.read(buffer.data(), file_size)) {
        std::cerr << "Error reading tf_graph file: " << checkpoint_filename << std::endl;
        return 1;
    }
    else {
        fprintf(stdout, "Successfully read tf_graph file\n");
    }

    // Create empty buffer and fill it with read in data
    TF_Buffer* tf_buffer = TF_NewBuffer();
    tf_buffer->data = buffer.data();
    tf_buffer->length = buffer.size();

    // Create empty graph
    TF_Graph* tf_graph = TF_NewGraph();
    TF_Status* status = TF_NewStatus();

    //File Graph with data from buffer
    TF_ImportGraphDefOptions* tf_graph_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(tf_graph, tf_buffer, tf_graph_opts, status);
    TF_DeleteImportGraphDefOptions(tf_graph_opts);
    TF_DeleteBuffer(tf_buffer);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error importing tf_graph: %s\n", TF_Message(status));
        TF_DeleteStatus(status);
        TF_DeleteGraph(tf_graph);
        return 1;
    }
    else { 
        fprintf(stdout, "Successfully imported graph: %s\n", TF_Message(status));
    }

    // Start a new TF_Session
    TF_SessionOptions* session_opts = TF_NewSessionOptions();
    TF_Session* session = TF_NewSession(tf_graph, session_opts, status);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error creating session: %s\n", TF_Message(status));
        TF_DeleteStatus(status);
        TF_DeleteGraph(tf_graph);
        return 1;
    }
    else {
        fprintf(stdout, "Successfully created session: %s\n", TF_Message(status));
    }


    // Get the input and output graph 
    TF_Operation* input_op = TF_GraphOperationByName(tf_graph, "images");
    if (input_op == nullptr) {
        fprintf(stderr, "Error: Failed to find operation 'images' in the graph.\n");
        TF_DeleteStatus(status);
        TF_DeleteGraph(tf_graph);
        return 1;
    }
    TF_Output input_var = {input_op, 0};
    TF_Operation* output_op = TF_GraphOperationByName(tf_graph, "features");
    if (output_op == nullptr) {
        fprintf(stderr, "Error: Failed to find operation 'features' in the graph.\n");
        TF_DeleteStatus(status);
        TF_DeleteGraph(tf_graph);
        return 1;
    }
    TF_Output output_var = {output_op, 0};
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error getting graph by name: %s\n", TF_Message(status));
        TF_DeleteStatus(status);
        TF_DeleteGraph(tf_graph);
        return 1;
    }

    // Get the output tensor shape and feature dimensions
    int64_t* dims = nullptr;
    int num_dims_out = TF_GraphGetTensorNumDims(tf_graph, output_var, status);
    if (num_dims_out < 0) {
        fprintf(stderr, "Error getting tensor dimensions for output_var\n");
        // Clean up and return error code
        TF_DeleteStatus(status);
        TF_DeleteSession(session, status);
        TF_DeleteGraph(tf_graph);
        return 1;
    }
    else if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Output is not in graph %s\n", TF_Message(status));
        TF_DeleteStatus(status);
        TF_DeleteGraph(tf_graph);
        return 1;
    }
    else {
        fprintf(stdout, "Successfully got tensor dimensions %s\n", TF_Message(status));
    }

    dims = new int64_t[num_dims_out];
    if (dims == nullptr) {
        fprintf(stderr, "Error allocating memory for dims\n");
        // Clean up and return error code
        TF_DeleteStatus(status);
        TF_DeleteSession(session, status);
        TF_DeleteGraph(tf_graph);
        return 1;
    }
    TF_GraphGetTensorShape(tf_graph, output_var, dims, num_dims_out, status);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error getting tensor shape for output_var: %s\n", TF_Message(status));
        // Clean up and return error code
        delete[] dims;
        TF_DeleteStatus(status);
        TF_DeleteSession(session, status);
        TF_DeleteGraph(tf_graph);
        return 1;
    }
    int feature_dim = dims[num_dims_out - 1];

    // Get the input tensor shape
    int64_t* image_shape = nullptr;
    int num_dims_in = TF_GraphGetTensorNumDims(tf_graph, input_var, status);
    if (num_dims_in < 0) {
        fprintf(stderr, "Error getting tensor dimensions for input_var\n");
        // Clean up and return error code
        delete[] dims;
        TF_DeleteStatus(status);
        TF_DeleteSession(session, status);
        TF_DeleteGraph(tf_graph);
        return 1;
    }
    image_shape = new int64_t[num_dims_in];
    if (image_shape == nullptr) {
        fprintf(stderr, "Error allocating memory for image_shape\n");
        // Clean up and return error code
        delete[] dims;
        TF_DeleteStatus(status);
        TF_DeleteSession(session, status);
        TF_DeleteGraph(tf_graph);
        return 1;
    }
    TF_GraphGetTensorShape(tf_graph, input_var, image_shape, num_dims_in, status);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error getting tensor shape for input_var: %s\n", TF_Message(status));
        // Clean up and return error code
        delete[] dims;
        delete[] image_shape;
        TF_DeleteStatus(status);
        TF_DeleteSession(session, status);
        TF_DeleteGraph(tf_graph);
        return 1;
    }

    // Check the rank of the input and output graph
    assert(num_dims_in == 4); // Check the number of dimensions
    assert(num_dims_out == 2); // Check the number of dimensions

    delete[] dims;
    delete[] image_shape;

    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(session_opts);
    TF_DeleteGraph(tf_graph);
    TF_DeleteStatus(status);

}
