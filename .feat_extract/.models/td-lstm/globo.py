import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]="3"


in_shapes = {
    "mobilenetv2":  (224, 224, 3),  #(224, 224, 3)
    "xception":     (299, 299, 3)     #(299, 299, 3)
}
BACKBONE = 'xception'  

CFG_TRAIN = {
    "backbone" : BACKBONE,
    "in_shape": in_shapes[BACKBONE],
    "in_height":in_shapes[BACKBONE][0],
    "in_width": in_shapes[BACKBONE][1],
}  


UCF101_ROOT = '/raid/DATASETS/anomaly/UCF101'
UCF101 = {
    'train_path' : UCF101_ROOT+'/train' ,
    'test_path' : UCF101_ROOT+'/test' ,
}