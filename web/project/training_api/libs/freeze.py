#===========================================================================================================================
# aims      :   Freeze the graph
#
# input     :   model_name : it takes the output_model_name set in 2_main_train_net.py
#
#               output_dir: it takes the train_output_net_path set in 2_main_train_net.py
#               output_node_names: it takes the softmax node name called "SMO" in 2_main_train_net.py
#
# return    :   void
#===========================================================================================================================

import tensorflow as tf

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


def freeze(model_name, output_dir, output_node_names):

    input_graph_path = output_dir + "/" + model_name + '.pbtxt'
    checkpoint_path = output_dir + "/" + model_name + '.ckpt'
    input_saver_def_path = ""
    input_binary = False

    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = output_dir + "/" + 'frozen_' + model_name + '.pb'
    output_optimized_graph_name = output_dir + "/" + 'optimized_' + model_name + '.pb'
    clear_devices = True

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_frozen_graph_name, clear_devices, "")

    # Optimize for inference
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(output_frozen_graph_name, 'rb') as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ["inputs/I"],  # an array of the input node(s)
        [output_node_names],  # an array of output nodes/softmax layer
        tf.float32.as_datatype_enum)

    # Save the optimized graph
    f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
    f.write(output_graph_def.SerializeToString())

    return {'grp_fr_name': output_frozen_graph_name, 'grp_opt_name': output_optimized_graph_name}


def load(pb_path):

    with tf.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # we import a graph_def into the current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph
