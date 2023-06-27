import tensorflow as tf
from tensorflow import keras

class ReshapeFeatures(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReshapeFeatures, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReshapeFeatures, self).build(input_shape)

    def call(self, x):
        return tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], -1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2] * input_shape[3] * input_shape[4])

    def get_config(self):
        config = super().get_config()
        return config


def all_operations(x):
    x_shape = tf.shape(x)
    batch_size, time_steps, height, width, channels = tf.split(x_shape, num_or_size_splits=5)
    tf.print("before all_opers", batch_size, time_steps, height, width, channels)
    x = tf.reshape(x, [batch_size[0], time_steps[0], height[0] * width[0] * channels[0]])
    tf.print("after all_opers", tf.shape(x))
    return x


def form_model(cfg):

    image_input = keras.Input(shape=(None,*cfg["in_shape"] ))

    if cfg["backbone"] == 'mobilenetv2':
        #original input_shape = (224, 224, 3)
        backbone = keras.applications.MobileNetV2(   include_top=True, \
                                                        weights='imagenet', \
                                                        input_shape = cfg["in_shape"])
    elif cfg["backbone"] == 'xception':
        #original input_shape = (299, 299, 3)
        backbone = keras.applications.Xception(  include_top=True, \
                                                    weights='imagenet', \
                                                    input_shape = cfg["in_shape"], \
                                                    )
    #print(backbone.summary())
    backbone = keras.models.Model(  inputs=backbone.layers[0].input,
                                    outputs=backbone.layers[-2].output)
    for layer in backbone.layers:
        layer.trainable=False
    
    tdbackbone_out = keras.layers.TimeDistributed(backbone)(image_input)
    
    #tf.print("BEFPRE ALL_OPER", tdbackbone_out.shape)
    
    #features_flatten = ReshapeFeatures()(tdbackbone_out)
    #tf.print("AFTER ALL_OPER", features_flatten.shape)
    # ( 1 , time_steps , spatl_featr_flattned ) 
    
    #features_shape = keras.backend.int_shape(features_flatten)
    #print(features_shape)
    #lstm_input_shape = (features_shape[1], features_shape[2])
    #lstm_input = keras.layers.Input(shape=lstm_input_shape)

    #lstm1 = keras.layers.LSTM(1024, return_sequences=False)(lstm_input)
    # ( 1 , time_steps , units) 
    
    #tdbackbone_spatial_flat = keras.layers.Lambda(all_operations)(tdbackbone_out)
    
    lstm2 = keras.layers.LSTM(1024,return_sequences=False,dropout=0.6)(tdbackbone_out)
    
    #global_rgb_feature = keras.layers.GlobalMaxPooling1D()(lstm1)
    # ( 1 , 1024 )

    mlp = keras.layers.Dense(512, activation='relu')(lstm2)
    mlp = keras.layers.Dropout(0.5)(mlp)
    mlp = keras.layers.Dense(128, activation='relu')(mlp)
    mlp = keras.layers.Dropout(0.5)(mlp)
    scores_out = keras.layers.Dense(cfg["out_nclasses"], activation='softmax')(mlp)

    model = keras.models.Model(inputs=image_input, outputs=scores_out)

    if cfg["optima"] == 'adam':
        optima = keras.optimizers.Adam(lr=cfg["lr"], decay=1e-6)

    model.compile(optimizer=optima, loss='categorical_crossentropy',
                metrics=[
                    keras.metrics.categorical_accuracy,
                    keras.metrics.top_k_categorical_accuracy
                ])
    model.summary()

    return model



if __name__ == "__main__":
    import globo , numpy as np , dataset

    model = form_model(globo.CFG_TRAIN)
    
    '''
    @tf.function
    def out(x):return model(x).numpy()
    
    train_data = dataset.DataGen(globo.ARGS.ds['train'],globo.CFG_TRAIN)
    train_tfds = dataset.create_tf_dataset(train_data,globo.CFG_TRAIN)
                                        
    for batch_index, (X, Y) in enumerate(train_tfds):
        print(f"Batch {batch_index}: X shape: {X.shape}, Y shape: {Y.shape}")
        output = model(X)
        print(output)
        break
    '''
    

    #data = np.ones(( globo.CFG_TRAIN["frame_max"] , *globo.CFG_TRAIN["in_shape"]),np.float32)
    #print(data.shape)
    #out = model(np.expand_dims(data , axis=0)).numpy()
    #print(out.shape)