import globo , model , dataset
import tensorflow as tf
from tensorflow import keras


if __name__ == "__main__":
    ## return model name based on cfg 
    model = model.form_model(globo.CFG_TRAIN)
    model_name = 'tdxception_ucf101_fs3'
    
    #data_train = dataset.VideoFrameGenerator(globo.UCF101['train_path'] , globo.CFG_TRAIN["backbone"])
    #data_test = dataset.VideoFrameGenerator(globo.UCF101['test_path'] , globo.CFG_TRAIN["backbone"])
    
    data_train = dataset.DataGen(globo.UCF101['train_path'] , globo.CFG_TRAIN["backbone"])
    data_test = dataset.DataGen(globo.UCF101['test_path'] , globo.CFG_TRAIN["backbone"])

    model.fit(data_train, epochs = 1, validation_data=data_test,
                    callbacks=[
                        keras.callbacks.ModelCheckpoint(
                            filepath="model/"+model_name+"_weights.{epoch:03d}.h5",
                            save_best_only=True,
                            monitor="val_categorical_accuracy",
                            save_freq='epoch',
                            verbose = 1
                        ),
                        keras.callbacks.CSVLogger(
                            filename="model/train_history.csv"
                        )
                    ],
                    use_multiprocessing = True , 
                    workers = 8 )
    
    model.save("model/"+model_name)
    model.save("model/"+model_name+".h5")