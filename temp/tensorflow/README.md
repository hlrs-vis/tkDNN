run 
$ g++ hello_tf.cpp -o hello_tf -ltensorflow -g
to compile

g++ hello_tf.cpp -o hello_tf -ltensorflow -g -o hello_tf `pkg-config --cflags --libs opencv4`