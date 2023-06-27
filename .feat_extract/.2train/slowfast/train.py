import os, time
import tensorflow as tf
from tensorflow.keras import backend as K
K.clear_session()

import matplotlib.pyplot as plt
import cv2
from src.utils.misc import get_precision, get_strategy, gpu_mem_usage
from src.utils.multigrid import MultigridSchedule

from src.models.builder import construct_model, compile_model
import src.models.zlowfat0 as zf
import src.models.x3d as x3d # Import the model definition to ensure it's registered

from src.data.loader import DSBuilder
from src.data.utils import bench_ds

import src.utils.log as log
logger = log.get_logger(__name__)



def train_epoch(model, optimizer, dataset, epoch):
    # Create a metric to track average loss per epoch
    epoch_loss_avg = tf.keras.metrics.Mean()
    
    # Iterate over the dataset batches
    for step, (inputs, labels) in enumerate(dataset):
        
        # Open a GradientTape to record the operations
        with tf.GradientTape() as tape:
            # Forward pass
            logits = model(inputs, training=True)

            # Compute the loss
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)

            # Calculate the gradients
            gradients = tape.gradient(loss, model.trainable_variables)

            # Apply the gradients using the optimizer
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Update the average loss metric
            epoch_loss_avg.update_state(loss)

        # Print the progress every few steps (e.g., every 10 steps)
        if step % 10 == 0:
            logger.info(f'Epoch {epoch + 1}, Step {step}, Loss: {epoch_loss_avg.result()}')

    # Return the average loss for the epoch
    return epoch_loss_avg.result()


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/cfg/default.py
    """

    if cfg.DEBUG:
        tf.config.experimental_run_functions_eagerly(True)
        tf.debugging.set_log_device_placement(True)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        tf.random.set_seed(1111)
        
        debug_path = os.path.join(cfg.EXPERIMENTPATH, "debug")
        if not os.path.exists(debug_path):os.mkdir(debug_path)
        logger.info(f'debub path @ {debug_path}')
        tf.debugging.experimental.enable_dump_debug_info(debug_path,
        tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)    


    ## precision
    precision = get_precision(cfg.TRAIN.MIXED_PRECISION)
    policy = tf.keras.mixed_precision.experimental.Policy(precision)
    tf.keras.mixed_precision.experimental.set_policy(policy)
    logger.info(f'train precision @ {precision} {policy}')
    
    
    ## Init multigrid.
    ## at the start of each epoch
    ##      if cfg.MULTIGRID.LONG_CYCLE:
    ##          cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
    ##              if changed:
    ##                  build_trainer with the new data parameters
    ##    Multigrid training is a mechanism to train video architectures efficiently. 
    ##    Instead of using a fixed batch size for training, 
    ##    this method proposes to use varying batch sizes in a defined schedule, 
    ##    yet keeping the computational budget approximately unchaged 
    ##    by keeping batch x time x height x width a constant. 
    ##    Hence, this follows a coarse-to-fine training process by having 
    ##    lower spatio-temporal resolutions at higher batch sizes and vice-versa. 
    ##    In contrast to conventioanl training with a fixed batch size, 
    ##    Multigrid training benefit from 'seeing' more inputs 
    ##    during a training schedule at approximately the same computaional budget.  
    '''
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    '''
    

    ## NOT IMPLEMENTED YET
    #strategy = get_strategy(cfg.GPUID)
    #with strategy.scope():
    
    
    train_ds = DSBuilder(cfg,'train')()
    print(train_ds)
    bench_ds(train_ds,100)
   
   
    steps_per_epoch = cfg.TRAIN.DATASET_SIZE // cfg.TRAIN.BATCH_SIZE
    #print(cfg.TRAIN.DATASET_SIZE , steps_per_epoch)
    
    
    model = construct_model(cfg)
    model , optima = compile_model(model,cfg,steps_per_epoch)
    
    
    # resume training from latest checkpoint, if available
    '''
    current_epoch = 0
    ckpt_path = tf.train.latest_checkpoint(model_dir)
    if ckpt_path:
      current_epoch = int(os.path.basename(ckpt_path).split('-')[1])
      logging.info(f'Found checkpoint {ckpt_path} at epoch {current_epoch}')
      model.load_weights(ckpt_path)
    elif FLAGS.pretrained_ckpt:
      logging.info(f'Loading model from pretrained weights at {FLAGS.pretrained_ckpt}')
      if tf.io.gfile.isdir(FLAGS.pretrained_ckpt):
        model.load_weights(tf.train.latest_checkpoint(FLAGS.pretrained_ckpt))
      else:
        model.load_weights(FLAGS.pretrained_ckpt)
    '''

    '''
    train_loss_results = []
    train_accuracy_results = []
    
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    def loss(model, x, y, training):
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        y_ = model(x, training=training)
        return loss_object(y_true=y, y_pred=y_)
    
    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    for epoch in range(cfg.SOLVER.MAX_EPOCH):
        
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        for x, y in train_ds:
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optima.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(y, model(x, training=True))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 1 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))
    '''
    
    '''
    model.fit(
        train_ds,
        verbose=1,
        epochs=1,#cfg.TRAIN.EPOCHS,
        initial_epoch = 0,
        steps_per_epoch=steps_per_epoch,
        #validation_data=get_dataset(cfg, FLAGS.val_file_pattern, False) if FLAGS.val_file_pattern else None,
        #callbacks=utils.get_callbacks(cfg, lr_schedule, FLAGS)
    )
    '''
    
    
    
    #logger.info(f'gpu mem usage {gpu_mem_usage()}')
    return None