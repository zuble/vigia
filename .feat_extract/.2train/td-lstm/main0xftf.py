import globo , model , dataset
import tensorflow as tf , time
from tensorflow import keras


if __name__ == "__main__":
    model = model.form_model(globo.CFG_TRAIN)
    #globo.init()
    
    train_tfds = dataset.create_tf_dataset('train')
    test_tfds = dataset.create_tf_dataset('test')

    t= time.time()
    model.fit(train_tfds, epochs = globo.ARGS.epochs, validation_data=test_tfds,
                    callbacks=[
                        keras.callbacks.ModelCheckpoint(
                            filepath=globo.WEIGHTS_PATH+"/ckpt-{epoch:03d}.h5",
                            save_best_only=True,
                            monitor="val_categorical_accuracy",
                            save_freq='epoch',
                            verbose = 1
                        ),
                        keras.callbacks.CSVLogger(
                            filename="model/train_history.csv"
                        )
                    ],
                    #use_multiprocessing = True , 
                    #workers = 8
                )
    tt=time.time()
    print("TRAIN IN",(str(tt-t)))
    
    print("\n SAVING MODEL .h5 @",globo.MODEL_PATH)
    if not globo.ARGS.dummy: 
        model.save_weights(globo.MODEL_PATH)