import globo
import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, roc_curve, precision_recall_curve , average_precision_score


DEBUG = False

def reshape_in(x):
    '''
    as the input format is cat( (bs_normal , ncrops , 32 , feat) , (bs_abnormal , ncrops , 32 , feat) )
    reshape so model processes all arrays of features, 1 array per crop per batch
    '''
    
    bs, ncrops, ts, feat = x.shape
    if DEBUG: tf.print("\nModelMultiCrop inputs = ", bs, ncrops, ts, feat)

    x = tf.reshape(x, (-1, ts, feat))  # ( bs * ncrops , ts , features)
    if DEBUG: 
        tf.print("inputs reshape = ", x.shape)

    return x, bs, ncrops


def reshape_out(x,bs,ncrops):
    '''
    as the output is a score for each bs*ncrops 32-feature_array 
    reshape so each crop from each batch_normal and abnormal is "exposed"
    calculate mean across all crops
    and get 32 scores per batch
    '''
    
    if DEBUG: print("scores = ", x.shape) ## (bs * ncrops , 32 ,1)
    
    ## get scores for each crop
    x = tf.reshape(x, (bs, ncrops, -1)) ## ( bs , ncrops , 32)
    if DEBUG: print("scores reshape = ", x.shape)
    
    ##########################################
    ## mean across the ncrops
    x = tf.reduce_mean(x, axis=1) ## (bs , 32)
    if DEBUG: print("scores mean = ", x.shape)
    ##########################################
    
    x = tf.expand_dims(x, axis=2) ## (bs , 32 , 1)
    if DEBUG: print("scores final = ", x.shape , x.dtype) 

    return x


def test_multicrop(model , normal_tfdata , abnormal_tfdata ):
    
    ## gt is in frame level , frist abormal then normal
    gt_all = np.load(globo.UCFCRIME_GT)
    total_frames = np.shape(gt_all)[0]
    
    scores_all = []
    frame_cnt = 0
    for i, data in enumerate(abnormal_tfdata):
        
        data = tf.expand_dims(data , 0) ## (1, ncrops , ts , 1024)
        data , bs , ncrops = reshape_in(data)
        scores = model(data)
        scores = reshape_out(scores, bs , ncrops)
        scores = tf.squeeze(scores,0)
        scores_exp = np.repeat(np.array(scores), 16) ## ts * 16 = original video frame state
        
        tsteps = np.shape(scores)[0]
        frames = tsteps * 16
        print(f'scores_exp {np.shape(scores_exp)}  , tsetps {tsteps} , frames {frames}')
        assert frames == np.shape(scores_exp)[0]
        
        scores_all.extend(scores_exp.tolist())
        frame_cnt += frames


    for i, data in enumerate(normal_tfdata):
        
        data = tf.expand_dims(data , 0) ## (1, ncrops , ts , 1024)
        #print("data",np.shape(data)) 
        
        data , bs , ncrops = reshape_in(data)
        scores = model(data)
        scores = reshape_out(scores, bs , ncrops)
        
        scores = tf.squeeze(scores,0)
        scores_expand = np.repeat(np.array(scores), 16) ## ts * 16 = original video frame state
        
        tsteps = np.shape(scores)[0]
        frames = tsteps * 16
        #print(f'scores_exp {np.shape(scores_expand)}  , tsetps {tsteps} , frames {frames}')
        assert frames == np.shape(scores_expand)[0]
        
        scores_all.extend(scores_expand.tolist())
        
        normal = np.zeros_like(scores_expand)
        print(f'scores_expand == correpondant interval gt ?\n{np.allclose(gt_all[frame_cnt : frame_cnt + frames] , normal)}')
        frame_cnt += frames
    

    print("\n\n",total_frames,np.shape(scores_all)[0])     
    if not globo.ARGS.dummy : assert total_frames == np.shape(scores_all)[0] == frame_cnt
    

    fpr_all, tpr_all, thresholds_all = roc_curve(gt_all, scores_all, pos_label=1)
    REC_AUC = auc(fpr_all, tpr_all)
    
    precision, recall, th = precision_recall_curve(gt_all, scores_all)
    PR_AUC = auc(recall, precision)

    AP = average_precision_score(gt_all, scores_all, pos_label=1)

    return REC_AUC , PR_AUC , AP


## deepmil
def test_deepmil(model, dataloader):
    model.eval()
    preds = []

    for input in dataloader:
        input_data = tf.reshape(input, [-1, input.shape[-1]])
        logits = model(input_data)
        logits = tf.squeeze(logits, 1)
        logits = tf.reduce_mean(logits, 0)
        sig = logits
        preds.append(sig)

    gt = np.load(globo.UCFCRIME_GT)
    preds = np.array(preds)
    preds = np.repeat(preds, 16)
    fpr, tpr, threshold = roc_curve(gt, preds)  # Calculate true positive rate and false positive rate
    np.save('.model/fpr.npy', fpr)
    np.save('.model/tpr.npy', tpr)
    rec_auc = auc(fpr, tpr)  # Calculate AUC value
    precision, recall, th = precision_recall_curve(gt, preds)
    pr_auc = auc(recall, precision)
    np.save('.model/precision.npy', precision)
    np.save('.model/recall.npy', recall)
    
    return rec_auc, pr_auc



if __name__ == "__main__":
    from loss import *
    from model import *
    from dataset import *
    
    model1 = ModelMultiCrop(globo.NFEATURES)
    
    test_normal_tfdata , test_abnormal_tfdata , niters = get_tfdataset(False)
    
    test(model1 , test_normal_tfdata , test_abnormal_tfdata )