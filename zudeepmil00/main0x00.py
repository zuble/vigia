import globo , os , datetime , time

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam , Adagrad

from model import *
from dataset import *
from loss import RankingLoss
from train import *
from test import test


    

if __name__ == "__main__":

    ## INIT MODEL FOLDER
    MODEL_NAME = "{:.4f}_{}_{}_{}_{}-{}".format(time.time(),globo.ARGS.classifier,globo.ARGS.features+globo.VERSION,globo.ARGS.lossfx,globo.OPTIMA,globo.LR)
    print(MODEL_NAME)
    BASE_MODEL_PATH = os.path.join('.model',MODEL_NAME)
    if not os.path.exists(BASE_MODEL_PATH):os.makedirs(BASE_MODEL_PATH);print("\nINIT MODEL FOLDER @",BASE_MODEL_PATH)
    else: raise Exception(f"{BASE_MODEL_PATH} eristes")
    WEIGHTS_PATH = os.path.join(BASE_MODEL_PATH,'weights'); os.makedirs(WEIGHTS_PATH)
    LOG_PATH = os.path.join(BASE_MODEL_PATH,'log'); os.makedirs(LOG_PATH)
    

    model = ModelMLP(globo.NFEATURES)
    
    if globo.OPTIMA == 'Adam':      optima = Adam( learning_rate=globo.LR ) #, weight_decay=0.00005
    elif globo.OPTIMA == 'Adagrad': optima = Adagrad( learning_rate=globo.LR ) 
    
    loss_obj = RankingLoss(lossfx = globo.ARGS.lossfx)
    
    
    train_normal_tfdata , train_abnormal_tfdata , niters = get_tfslices()
    test_normal_dataset , test_abnormal_dataset , niters = get_tfslices(False)
    
        
    for epoch in range(globo.ARGS.epochs):
        
        losses = train_gen( model, \
                        train_normal_tfdata, train_abnormal_tfdata, \
                        niters , \
                        optima, loss_obj , globo.NCROPS)
        
        print(f'\n\nEPOCH {epoch + 1}/{ globo.ARGS.epochs} , Average Loss: {np.mean(losses):.4f}\n\n') 

        if (epoch + 1) % 2 == 0 and not globo.ARGS.dummy:
            model.save_weights(os.path.join(WEIGHTS_PATH, f'{MODEL_NAME}_EP-{epoch + 1}.h5'))   
            rec_auc , pr_auc , ap = test(model, test_normal_dataset , test_abnormal_dataset)
            print('\nTEST rec_auc = {} , pr_auc = {} , ap = {}'.format(rec_auc,pr_auc,ap),globo.NCROPS)

    ## https://www.tensorflow.org/guide/keras/save_and_serialize
    print("\n SAVING MODEL .h5 @",BASE_MODEL_PATH+'/'+MODEL_NAME)
    model.save_weights(BASE_MODEL_PATH+'/'+MODEL_NAME+'.h5')

    '''
    from tensorflow.keras.models import load_model
    loaded_model = load_model(os.path.join(MODEL_PATH, 'saved_model'))
    
    ## as the model is constructed in tf.keras.model i should use this 
    
    nfeatures = globo.NFEATURES
    loaded_model = ModelMultiCrop(nfeatures)
    loaded_model.build((None, nfeatures))  # Build the model with the proper input shape
    loaded_model.load_weights("model_weights.h5")
    '''