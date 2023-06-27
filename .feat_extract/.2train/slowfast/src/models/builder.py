import tensorflow as tf

from src.utils.optima import get_optima

# Model registry
MODEL_REGISTRY = {}

# Model registration function
def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def construct_model(cfg):
    ## Construct the model
    ## x3d is constructed as keras.model
    ## slowfast doenst has a forward/call fx
    name = cfg.MODEL.MODEL_NAME
    if name == 'SlowFastRes50':
        model_instance = MODEL_REGISTRY.get(name)(cfg)
        model = model_instance.model
    else:
        model= MODEL_REGISTRY.get(name)(cfg)
    
    if cfg.LOG_MODEL_INFO: model.summary()
    return model


def compile_model(model,cfg,steps_per_epoch=16):
    ''''''
    optima = get_optima(cfg,steps_per_epoch)

    model.compile(
        optimizer=optima,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=5,
                name='top_5_acc')])

    return model , optima