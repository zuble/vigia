import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import keras.backend as K

import utils.tf_formh5 as tf_formh5
import utils.auxua as aux
import logging , os


''' GPU CONFIGURATION '''
tf_formh5.set_tf_loglevel(logging.ERROR)
tf_formh5.tf.debugging.set_log_device_placement(False) #Enabling device placement logging causes any Tensor allocations or operations to be printed.
tf_formh5.set_memory_growth()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]="1"


''' INIT TEST MODEL '''
wght4test_config = {
    "ativa" : 'relu',
    "optima" : 'sgd',
    "batch_type" : 0, # =0 all batch have frame_max or video length // =1 last batch has frame_max frames // =2 last batch has no repetead frames
    "frame_max" : '4000' 
}


model, model_name = tf_formh5.init_test_model(wght4test_config,from_path=aux.WEIGHTS_PATH)

# to get .pb file model
h5_pb_path = '/raid/DATASETS/.zuble/vigia/zurgb/model/model_trt/relu_sgd_0_4000/pb'
tf.saved_model.save(model, h5_pb_path)


trt_path = '/raid/DATASETS/.zuble/vigia/zurgb/model/model_trt/relu_sgd_0_4000/trt' # define path to save model in saved model format

# https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#usage-example
print("\nconverting to trt-model")
# we feed the saved model into this converter function
converter = trt.TrtGraphConverterV2(input_saved_model_dir=h5_pb_path )

converter.experimental_new_converter = True

print("\nconverter.convert")
converter.convert()

print("\nconverter.save")
converter.save(trt_path) # we save the converted model

print("trt-model saved under: ",trt_path)




'''tf.__version__

dir = '/raid/DATASETS/.zuble/vigia/zurgb/model/model_trt/relu_sgd_0_4000'
pb_dir =  dir+'/pb'

def frozen_keras_graph(model):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config

    real_model = tf.function(model).get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

    input_tensors = [
        tensor for tensor in frozen_func.inputs
        if tensor.dtype != tf.resource
    ]
    output_tensors = frozen_func.outputs

    graph_def = run_graph_optimizations(
        graph_def,
        input_tensors,
        output_tensors,
        config=get_grappler_config(["constfold", "function"]),
        graph=frozen_func.graph)
    
    return graph_def
  

model = tf_formh5.form_model(wght4test_config)

graph_def = frozen_keras_graph(model)
    
# frozen_func.graph.as_graph_def()
tf.io.write_graph(graph_def, pb_dir , 'relu_sgd_0_4000_frozen_graph')'''