import globo , csv , os , datetime , time

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam , Adagrad

from model import *
from dataset import *
from loss import RankingLoss
from train import *
from test import test

    

if __name__ == "__main__":


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
            if not globo.ARGS.druna: 
                model.save_weights(os.path.join(globo.WEIGHTS_PATH, f'{globo.MODEL_NAME}_EP-{epoch + 1}.h5'))   
            
            auc , pr_auc , ap , fpr , tpr , th_targ = test(model, test_normal_dataset , test_abnormal_dataset , globo.NCROPS)
            print('\nTEST roc_auc = {} , pr_auc = {} , ap = {} , fpr = {} , tpr = {} , th_targ = {}'.format(auc,pr_auc,ap,fpr,tpr,th_targ))
        else: auc , pr_auc , ap , fpr , tpr , th_targ = '', '', '' , '' , '' , ''
        
        
        if not globo.ARGS.druna: 
            globo.histlog([epoch + 1, np.mean(losses), auc , pr_auc , ap , fpr , tpr , th_targ])
       
            
    ## https://www.tensorflow.org/guide/keras/save_and_serialize
    print("\n SAVING MODEL .h5 @",globo.MODEL_PATH)
    if not globo.ARGS.druna: 
        model.save_weights(globo.MODEL_PATH)

    '''
    from tensorflow.keras.models import load_model
    loaded_model = load_model(os.path.join(MODEL_PATH, 'saved_model'))
    
    ## as the model is constructed in tf.keras.model i should use this 
    
    nfeatures = globo.NFEATURES
    loaded_model = ModelMultiCrop(nfeatures)
    loaded_model.build((None, nfeatures))  # Build the model with the proper input shape
    loaded_model.load_weights("model_weights.h5")
    '''