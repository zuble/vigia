from utils import globo ,  xdv , tfh5 , rslts

import os, random, cv2 , csv , time

#import matplotlib.pyplot as plt
import numpy as np
from tqdm.keras import TqdmCallback

import tensorflow as tf
tf.debugging.set_log_device_placement(False) #Enabling device placement logging causes any Tensor allocations or operations to be printed.
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input


class DataTest():   
    def __init__(self, model):
        
        self.mode = 'test'
        self.model = model
        self.in_height = globo.CFG_WEIGHTS4TEST["in_height"]
        self.in_width = globo.CFG_WEIGHTS4TEST["in_width"]
        
        ## preprocessing fx
        if globo.CFG_WEIGHTS4TEST["backbone"] == "original":
            self.prep_fx = self.normalize
        elif globo.CFG_WEIGHTS4TEST["backbone"] == "mobilenetv2":
            self.prep_fx = mobilenet_v2_preprocess_input
        elif globo.CFG_WEIGHTS4TEST["backbone"] == "xception":
            self.prep_fx = xception_preprocess_input
        
        print(globo.CFG_RGB_TEST)
        self.batch_type = globo.CFG_RGB_TEST["batch_type"]
        self.frame_max = globo.CFG_RGB_TEST["frame_max"]
        self.watch = globo.CFG_RGB_TEST["watch"]
        self.debug = globo.CFG_RGB_TEST["debug"]
        self.dummy = globo.CFG_RGB_TEST["dummy"]
        
        #npz_path = os.path.join(globo.AAS_PATH+"/2_sub_interval",f"{globo.CFG_SINET['sinet_version']}--fl2_{self.mode}-{globo.CFG_SINET['chunck_fsize']}fi.npz")
        #if not os.path.exists(npz_path):
        #    print("npz no existe:", npz_path)
        #    return 
        #data = np.load(npz_path, allow_pickle=True)["data"]
        
        vpaths , labels , tframes = xdv.load_test_npy()
        if self.dummy:
            self.vpaths = vpaths[:self.dummy]
            self.labels = labels[:self.dummy]
            self.tframes = tframes[:self.dummy]
            self.len_data = self.dummy
        else:
            self.vpaths = vpaths
            self.labels = labels
            self.tframes = tframes
            self.len_data = len(self.vpaths)
        print("\n\nDataGen",self.mode,self.len_data,\
            "\n\tNORMAL intervals", sum(1 for label in labels if label == 0 ),\
            "\n\tABNORMAL intervals", sum(1 for label in labels if label == 1 ),"\n\n")

        ## watch stuff
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.5;self.thickness = 1;self.lineType = cv2.LINE_AA
        self.green = (0, 255, 0);self.red = (0, 0, 255)
    
    
    def normalize(self,x): return x/255.0
    
    
    def watch_this_interval(self,idx,aas):
        vpath = self.vpaths[idx]
        sf = self.data[idx]['sf']
        ef = self.data[idx]['ef']
        label = self.labels[idx]
        
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print("Error opening video file")
            return

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:break

            frame_count += 1
            if sf <= frame_count <= ef:
                cv2.putText(frame, str(label), (10, 30), self.font, self.fontScale, self.green, self.thickness, self.lineType)
                cv2.putText(frame, str(aas), (10, 60), self.font, self.fontScale, self.red, self.thickness, self.lineType)
                cv2.imshow('Video', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):break

        cap.release()
        cv2.destroyAllWindows()
        

    def extract_frame_batches(self, vpath):
        cap = cv2.VideoCapture(vpath)
        frame_batches = []
        batch_frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.resize(frame, (self.in_width, self.in_height))
            frame_rgb = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))#/255.0
            frame_prep = self.prep_fx(frame_rgb)    ## uint8 -> float32
            batch_frames.append(frame_prep)
            
            if len(batch_frames) == self.frame_max:
                #print("batching", np.shape(np.array(batch_frames)))
                frame_batches.append(np.array(batch_frames))
                batch_frames = []

        # Store any remaining frames in the last batch
        #if len(batch_frames) > 0:
        #    frame_batches.append(np.array(batch_frames))

        cap.release()
        print("end batching", np.shape(frame_batches))
        return frame_batches
        
    
    @tf.function
    def aas_predict(self,x):
        return self.model(x, training=False)    

    def run(self):
        
        aas_list = []
        for idx in range(self.len_data):
            vpath = self.vpaths[idx]
            label = self.labels[idx]
            tframe = self.tframes[idx]
            
            print("\n################ START test.run idx",idx,"##############",\
                "\n\t",vpath)
            
            frame_batches = self.extract_frame_batches(vpath)
            predictions = []
            for batch_frames in frame_batches:
                t0 = time.time()
                tf_out = self.aas_predict(np.expand_dims(batch_frames, 0))
                tf_out = tf.constant(tf_out)
                aas = tf_out.numpy().item()
                t1 = time.time()
                print(aas,t1-t0)
                #predictions.extend(aas)
                
            #print(np.max(np.array(predictions)),t1-t0)
            aas_list.append(aas) 
            
            if self.debug : print(  "\n",idx,os.path.basename(vpath),\
                                    "\nlabel",label,\
                                    "\naas",aas,\
                                    '\nin',format(t1-t0,f".{3}f"),"secs",\
                                    "\n################ END test.run idx",idx,"##############")
        
        
        #label_list = [ self.data[i]["label"] for i in range(self.len_data)]
        #assert len(aas_list) == len(label_list)

        #if self.watch:
        #    for idz in range(self.len_data):
        #        label = label_list[idz] 
        #        aas = aas_list[idz]
        #        if (label == 0 and aas > 0.5) or (label == 1 and aas < 0.5):
        #            self.watch_this_interval(idz,aas)
        #    
        #rslts.get_cm_accuracy_precision_recall_f1(model_name,label_list,aas_list,plot=True)

        
if __name__ == "__main__":
    
    model_name = '1684023431.828157_mobilenetv2_relu_adam-0.0002_0_4_4000'
    model_path = globo.MODEL_PATH + model_name
    model = tf.keras.models.load_model(model_path)
    
    test = DataTest(model)
    test.run()













''' TEST FX '''
'''
@tf.function
def vas_predict(model,x):
    return model(x, training=False)    

def input_test_video_data(file_name,config,batch_no=0):
    video = cv2.VideoCapture(file_name)
    total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    divid_no = 1
    frame_max = config["frame_max"]
    batch_type = config["batch_type"]
    
    # define the nmbers of batchs to divid atual video (divid_no)
    if total_frame > int(frame_max):
        total_frame_int = int(total_frame)
        if total_frame_int % int(frame_max) == 0:
            divid_no = int(total_frame / int(frame_max))
        else:
            divid_no = int(total_frame / int(frame_max)) + 1


    #updates the start frame to 0,frame_max*1,frame_max*2... excluding the last batch
    passby = 0
    if batch_no != divid_no - 1:
        while video.isOpened and passby < int(frame_max) * batch_no:
            passby += 1
            success, image = video.read()
            if success == False:
                break
            
    #updates the last batch starting frame 
    else:
        if batch_type==1:
            #print("1")
            while video.isOpened and passby < total_frame - int(frame_max):
                passby += 1
                success, image = video.read()
                if success == False:
                    break
        #last batch must have >= frame_max/10 otherwise it falls back to batch_type 1
        if batch_type==2 and total_frame - (int(frame_max) * batch_no) >= int(frame_max)*0.1:
            #print("2")
            while video.isOpened and passby < int(frame_max) * batch_no:
                passby += 1
                success, image = video.read()
                if success == False:
                    break
        else:
            while video.isOpened and passby < total_frame - int(frame_max):
                passby += 1
                success, image = video.read()
                if success == False:
                    break

            
    batch_frames, batch_imgs = [], []
    counter = 0
    
    while video.isOpened:               
        success, image = video.read()
        if success == False:
            break
        batch_imgs.append(image)
        
        image_rsz = cv2.resize(image, (in_width, in_height))
        image_array = np.array(image_rsz)/255.0 #normalize
        batch_frames.append(image_array)
        
        counter += 1
        if counter > int(frame_max):
            break
            
    video.release()
    
    #batch_frames_tensor = tf.convert_to_tensor(batch_frames)
    ##print("\tshap tensor",tf.shape( tf.expand_dims(batch_frames_tensor,0) ) )
    #print("\t-batch",batch_no,"[",passby,", ... ] ", batch_frames_tensor.get_shape().as_list() )    

    batch_frames = np.array(batch_frames)
    print("\t-batch",batch_no,"[",passby,", ... ] ",batch_frames.shape)    

    return np.expand_dims(batch_frames,0), batch_imgs, divid_no, total_frame, passby, fps

def test_model(model,model_name,config,files=test_fn):
    print("\n\nTEST MODEL\n")

    # rslt txt file creation
    txt_path = globo.RSLT_PATH+model_name+'-'+str(config["batch_type"])+'_'+str(config["frame_max"])+'.txt'
    if os.path.isfile(txt_path):raise Exception(txt_path,"eriste")
    else: print("\tSaving @",txt_path,"\n")
    
    f = open(txt_path, 'w')
    
    content_str = ''
    total_frames_test = 0
    
    predict_total = [] #to output predict in vizualizer accordingly to the each batch prediction
    predict_max = 0     #to print the max predict related to the file in test
    predict_total_max = [] #to perform the metrics
    
    start_test = time.time()
    for i in range(len(files)):
        if files[i] != '':
            file_path = files[i]
            predict_result = () #to save predictions per file
            time_batch_predict = time_video_predict = 0.0

            #the frist 4000 frames from actual test video                
            batch_frames, batch_imgs, divid_no, total_frames,start_frame, fps = input_test_video_data(file_path,config)
            video_time = total_frames/fps
            total_frames_test += total_frames
            
            #prediction on frist batch
            start_predict1 = time.time()
            #predict_aux = model.predict(batch_frames)[0][0]
            
            predict_aux = vas_predict(model,batch_frames)[0][0].numpy() #using tf.function
            #predict_aux = model(batch_frames,training=False)[0][0]
            end_predict1 = time.time()
            time_video_predict = time_batch_predict = end_predict1-start_predict1
            
            predict_max = predict_aux
            predict_result = (divid_no,start_frame+batch_frames.shape[1],predict_max)
            #print(predict_result,batch_frames.shape)
            
            high_score_patch = 0
            print("\t ",predict_max,"%"," in ","{:.4f}".format(time_batch_predict)," secs")
            
            #when batch_frames (input video) has > frame_max frames
            patch_num = 1
            while patch_num < divid_no:
                batch_frames, batch_imgs, divid_no, total_frames,start_frame, fps = input_test_video_data(file_path,config,patch_num)
                
                #nésimo batch prediction
                start_predict2 = time.time()
                #predict_aux = model.predict(batch_frames)[0][0]
                predict_aux = vas_predict(model,batch_frames)[0][0].numpy() #using tf.function
                end_predict2 = time.time()
                time_batch_predict = end_predict2 - start_predict2
                time_video_predict += time_batch_predict

                if predict_aux > predict_max:
                    predict_max = predict_aux
                    high_score_patch = patch_num
                
                predict_result += (start_frame,start_frame+batch_frames.shape[1], predict_aux)
                #print(predict_result)
                
                print("\t ",predict_aux,"%"," in ","{:.4f}".format(time_batch_predict)," secs")  
                patch_num += 1
            
            predict_total.append(predict_result)
            predict_total_max.append(predict_max)
            print("\n\t",predict_total[i])
            
            if 'label_A' in files[i]:
                print('\nNORM',str(i),':',f'{total_frames:.0f}',"@",f'{fps:.0f}',"fps =",f'{video_time:.2f}',"sec\n\t",
                        files[i][files[i].rindex('/')+1:],
                        "\n\t "+str(predict_max),"% @batch",high_score_patch,"in",str(time_video_predict),"seconds\n",
                        "----------------------------------------------------\n")
            else:
                print('\nABNORM',str(i),':',f'{total_frames:.0f}',"@",f'{fps:.0f}',"fps =",f'{video_time:.2f}',"sec\n\t",
                        files[i][files[i].rindex('/')+1:],
                        "\n\t"+str(predict_max),"% @batch",high_score_patch,"in",str(time_video_predict),"seconds\n",
                        "----------------------------------------------------\n")
                
            content_str += files[i][files[i].rindex('/')+1:] + '|' + str(predict_total_max[i]) + '|' + str(predict_total[i])  + '\n'
            
    end_test = time.time()
    time_test = end_test - start_test

    f.write(content_str)
    f.close()
    print("\nDONE\n\ttotal of",str(total_frames_test),"frames processed in",time_test," seconds",
            "\n\t"+str(total_frames_test / time_test),"frames per second",
            "\n\n********************************************************",
            "\n\n********************************************************")                  

    #remove white spaces in file , for further easier reading
    with open(txt_path, 'r+') as f:txt=f.read().replace(' ', '');f.seek(0);f.write(txt);f.truncate()
    #aux.sort_files(txt_path) #sort alphabetcly #also done in get_rslts_from_txt
    
    return predict_total_max, predict_total

'''