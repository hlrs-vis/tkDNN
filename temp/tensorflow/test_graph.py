import tensorflow as tf

# Load the graph
with tf.io.gfile.GFile('mars-small128.pb', 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# Print the names of all operations in the graph
for op in graph_def.node:
    print(op.name)