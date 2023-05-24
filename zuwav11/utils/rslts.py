
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix , accuracy_score , f1_score , precision_score , average_precision_score , roc_auc_score , recall_score
from keras import backend as K


#--------------------------------------------------------#
# RESULTS PRINT AND PLOTS

""" 
METRICS/RESULTS CALCULUS
https://www.tensorflow.org/tutorials/structured_data/imbalanced_data 
"""

def plot_cm(name,labels,predictions,threshold=0.5,save=False):
    '''
    https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#download_the_kaggle_credit_card_fraud_data_set
    '''
    predictions = np.array(predictions)
    cm = confusion_matrix(labels, predictions > threshold)
    #plt.clf()
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(threshold))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    if save: plt.savefig(name+'.png',facecolor='white', transparent=False)
    plt.show()

def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, color=colors[0], linestyle='--')
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,80])
    plt.ylim([20,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig(name+'.png',facecolor='white', transparent=False)
    plt.show()
    
def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)
    #plt.clf()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.plot(precision, recall, label=name, linewidth=2, color=colors[0])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.xlim([-0.5,100.5])
    plt.ylim([-0.5,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig(name+'.png',facecolor='white', transparent=False)
    plt.show()

def get_cm_accuracy_precision_recall_f1(model_name,labels, predictions,printt=True,plot=False,save_plot=False):
    
    print(  "\n\n\n££££££££££££££££££££££££££££££££££££££££££££££££££££££",\
            "\n\n\tCM_ACC_PRE_REC_F1 4",model_name,"\n")
    predictions_binary = [1 if p >= 0.5 else 0 for p in predictions]
    
    if plot:
        plot_cm(str(model_name),labels, predictions_binary, save=save_plot)
    
    #acc = tf.metrics.BinaryAccuracy()
    #acc.update_state(labels, predictions)
    #acc_res = acc.result().numpy()
    acc_res = accuracy_score(labels, predictions_binary)
    if printt:print("\tACCURACY  %.4f"% acc_res)

    #pre = tf.keras.metrics.Precision(thresholds = 0.5)
    #pre.update_state(labels, predictions)
    #pre_res = pre.result().numpy()
    pre_res = precision_score(labels, predictions_binary)
    if printt:print("\tPRECISION (%% of True 1 out of all Positive predicted) %.4f"% pre_res)
    
    #rec = tf.keras.metrics.Recall(thresholds=0.5)
    #rec.update_state(labels, predictions)
    #rec_res = rec.result().numpy()
    rec_res = recall_score(labels, predictions_binary)
    if printt:print("\tRECALL (%% of True Positive out of all actual anomalies) ",rec_res)
    
    #https://glassboxmedicine.com/2019/03/02/measuring-performance-auprc/
    auprc_ap = average_precision_score(labels, predictions)
    aucroc = roc_auc_score(labels, predictions)
    if printt:print("\tAUPRC ( AreaUnderPrecisionRecallCurve ) %.4f \n\t AUC-ROC %.4f "% (auprc_ap, aucroc))
    
    #https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score
    #import tensorflow_addons as tfa
    #f1 = tfa.metrics.F1Score(num_classes=2, threshold=0.5)
    #f1.update_state(labels, predictions)
    f1_res1 = f1_score(labels, predictions_binary)
    f1_res2 = 2*((pre_res*rec_res)/(pre_res+rec_res+K.epsilon()))
    if printt:print("\tF1_SCORE (harmonic mean of precision and recall)  %.4f %.4f "% (f1_res1,f1_res2))
    
    return (acc_res,pre_res,rec_res,auprc_ap,aucroc,f1_res1)
