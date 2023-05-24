import os

import matplotlib.pyplot as plt
import pandas as pd

from utils import globo
#------------------------------------------------------------#
# hist.csv

def count_files(directory):
    return len([file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))])

def strip_model_fn(fn):
    tokens = fn.split("_")[1:]
    return str(tokens)

def get_histplot_from_csv(fldr_or_file=globo.HIST_PATH,versus=False,save=False,show=True):
    
    strings=["loss","accuracy","precision","recall","auc","prc","tp","fp","tn","fn",\
            "val_loss","val_accuracy","val_precision","val_recall","val_auc","val_prc","val_tp","val_fp","val_tn","val_fn"]
    csv_path=[]
    csv_fn=[]
    plt.style.use('bmh')
    
    if not versus:

        if os.path.isfile(fldr_or_file):
            fname, fext = os.path.splitext(fldr_or_file)
            print(fname)
            csv_path.append(os.path.join(fldr_or_file))
            csv_fn.append(fname)
        else:
            for root, dirs, files in os.walk(fldr_or_file):
                for file in files:    
                    fname, fext = os.path.splitext(file)
                    if fext == ".csv":
                        print(fname)
                        csv_path.append(os.path.join(fldr_or_file,file))
                        csv_fn.append(fname)
                break

        #1 PLOT loss+val_loss / acc+val_acc /... PER MODEL
        for csv in range(len(csv_path)):
            data = pd.read_csv(csv_path[csv]) # read csv file
            print(csv)
            #single metrics per plot
            for i in range(0,6):
                plt.plot(data[strings[i]],label=strings[i])
                plt.plot(data[strings[i+10]],label=strings[i+10]) # 4validation
                plt.xlabel('epochs');plt.ylabel(strings[i])
                plt.legend();plt.title(csv_fn[csv])
                plt.show()

            # tp,fn,tn,fn + it's val all in 1 plot
            strs=[]
            for j in range(6,10):
                plt.plot(data[strings[j]]);strs.append(strings[j])
                plt.plot(data[strings[j+10]]);strs.append(strings[j+10])
            plt.xlabel('epochs')
            plt.ylabel('videos')
            plt.legend(strs)
            plt.title(csv_fn[csv])
            plt.show()


    # PLOT SAME METRICS TOGETHER FOR ALL MODEL HISTORY .csv
    else:
        if os.path.isfile(fldr_or_file): 
            raise Exception("must be folder to print the metrics versus per model")
        elif count_files(fldr_or_file) == 1:
            raise Exception("must have more than 1 history to print the metrics versus per model")
        
        for root, dirs, files in os.walk(fldr_or_file):
            for file in files:    
                fname, fext = os.path.splitext(file)
                if fext == ".csv":
                    print(fname)
                    csv_path.append(os.path.join(fldr_or_file,file))
                    csv_fn.append(fname)
            break

        for i in list(range(0,2)) + list(range(10,16)):
            for csv in range(len(csv_path)):  
                data = pd.read_csv(csv_path[csv])
                label = strip_model_fn(csv_fn[csv])
                #print(csv_path[csv]+"\n"+label)
                plt.plot(data[strings[i]],label=label)
              
            plt.xlabel('epochs');plt.ylabel(strings[i])
            plt.legend();plt.title(strings[i]+' VS')
            
            if save: plt.savefig(os .path.join(os.path.dirname(csv_path[0]),strings[i]+'.png'))
            if show:plt.show()
      
            
#--------------------------------------------------------#
# RESULTS PRINT AND PLOTS        

import seaborn as sns
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix , accuracy_score , f1_score , precision_score , average_precision_score , roc_auc_score , recall_score
from keras import backend as K

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