import tensorflow as tf
from tensorflow import keras


def all_operations(args):
    x = args[0] ; #tf.print(x.shape)
    x = tf.reshape(x, [1, -1,x.shape[1]*x.shape[2]*x.shape[3]])
    return x

def form_model(cfg):

    image_input = keras.Input(shape=(None, *cfg["in_shape"] ))

    if cfg["backbone"] == 'mobilenetv2':
        #original input_shape = (224, 224, 3)
        backbone = tf.keras.applications.MobileNetV2(   include_top=False, \
                                                        weights='imagenet', \
                                                        input_shape = cfg["in_shape"])
        tdbackbone_out = keras.layers.TimeDistributed(backbone)(image_input)
    elif cfg["backbone"] == 'xception':
        #original input_shape = (299, 299, 3)
        backbone = tf.keras.applications.Xception(  include_top=False, \
                                                    weights='imagenet', \
                                                    input_shape = cfg["in_shape"], \
                                                    pooling=None)
        tdbackbone_out = keras.layers.TimeDistributed(backbone)(image_input)

    for layer in backbone.layers:
        layer.trainable=False

    features_flatten = keras.layers.Lambda(all_operations)(tdbackbone_out)  # flatten spatial features to time series
    # ( 1 , time_steps , spatl_featr_flattned ) 
    lstm1 = keras.layers.LSTM(1024, return_sequences=True)(features_flatten) #input_shape=(120,c3d_mp_flatten.shape[2]),
    # ( 1 , time_steps , units) 
    global_rgb_feature = keras.layers.GlobalMaxPooling1D()(lstm1)
    # ( 1 , 1024 ) 

    mlp = keras.layers.Dense(512, activation='relu')(global_rgb_feature)
    mlp = keras.layers.Dropout(0.5)(mlp)
    mlp = keras.layers.Dense(128, activation='relu')(mlp)
    mlp = keras.layers.Dropout(0.5)(mlp)
    scores_out = keras.layers.Dense(101, activation='softmax')(mlp)

    model = keras.models.Model(inputs=image_input, outputs=scores_out)

    optima = keras.optimizers.Adam(lr=1e-5, decay=1e-6)

    model.compile(optimizer=optima, loss='categorical_crossentropy',
                metrics=[
                    keras.metrics.categorical_accuracy,
                    keras.metrics.top_k_categorical_accuracy
                ])
    model.summary()

    return model

if __name__ == "__main__":
    import globo
    model = form_model(globo.CFG_TRAIN)
