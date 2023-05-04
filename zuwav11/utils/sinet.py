import json , time, subprocess , os , sys

import numpy as np

import moviepy.editor as mp
#import essentia 
#print(essentia.__version__)
#print(essentia.__file__)
import essentia.standard as es

    
class Sinet:
    def __init__(self, model_config):

        self.model_config = model_config
        
        ## MODEL & METADATA
        self.model = es.TensorflowPredictFSDSINet(
                            graphFilename=self.model_config['graph_filename'],
                            batchSize=self.model_config["batchSize"],
                            lastPatchMode=self.model_config["lastPatchMode"],
                            patchHopSize=self.model_config["patchHopSize"] )
        self.metadata = json.load(open(self.model_config['metadata_file'], "r"))
        
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        print("\n\nSINET VERSION", self.model_config["sinet_version"] , "w",self.model_config['full_or_max'],"output")
        
    def print_acodec_from_mp4(self, data, printt=False, only_sr=False):
        out = []
        for i in range(len(data)):
            output = subprocess.check_output('ffprobe -hide_banner -v error -select_streams a:0 -show_entries stream=codec_name,channels,sample_rate,bit_rate -of default=noprint_wrappers=1 -of compact=p=0:nk=1 ' + str('"' + data[i] + '"'), shell=True)
            output = str(output).replace("\\n", "").replace("b", "").replace("'", "").splitlines()[0]
            out.append(output)
            if printt: print(output)
        if only_sr: return int(str(out[0]).split('|')[1])
        else: return out

    def get_sigmoid(self, vpath , sf = 0 , ef = -1 , debug = False):
        '''
            gets sigmid from a interval if sf/ef defined 
            or simply from full audio
        '''
            
        ## FS converter
        mp4_fs_aac = self.print_acodec_from_mp4([vpath], only_sr=True)
        resampler = es.Resample(inputSampleRate=mp4_fs_aac, outputSampleRate=self.model_config['audio_fs_input'])
        
        if debug: tt1 = time.time()
        
        audio_es = mp.AudioFileClip(filename=vpath, fps=mp4_fs_aac)
        # cuts the video if sf / tf different than abnormal flag(normal video have lengthy duration)
        if sf != 0 and ef != -1:
            st = sf / 24; et = ef / 24
            audio_es_cut = audio_es.subclip(t_start=st,t_end=et)
            aud_arr = audio_es_cut.to_soundarray()
        else: aud_arr = audio_es.to_soundarray()
        aud_arr_mono_single = np.mean(aud_arr, axis=1).astype(np.float32)
        aud_arr_essentia = resampler(aud_arr_mono_single)

        ## predict
        p_es = self.model(aud_arr_essentia)
                
        audio_es.close() 
        if debug: 
            tt2 = time.time()
            print(  "\n\n************** START sinet.get_sigmoid ***************",\
                    "\n\t",os.path.basename(vpath)," w/ ", mp4_fs_aac ,"hZ",\
                    "\n\t",sf, ef,\
                    "\n\t MAX aas for anom labels 2")
            for i in range(len(self.model_config['anom_labels_i2'])):
                label_i = self.model_config['anom_labels_i2'][i]
                print("\t",label_i, self.model_config['anom_labels2'][i], np.amax(np.asarray(p_es)[:, label_i]))
            print('\n\n\t p_es @ sinet', np.shape(p_es),'@',str(tt2 - tt1),"secs") #"expls", np.amax(p_es[:, 72])
            print("\n***************** END sinet.get_sigmoid ********************")
            
        if self.model_config['full_or_max'] == 'full': return p_es
        elif self.model_config['full_or_max'] == 'max': return np.max(p_es, axis=0)
        
        
    def get_sigmoid_fl(self, vpath , frame_intervals , debug = False):
        '''
            gets 1 full sigmoid for each pair of (start_frame,eend_frame)
        '''
        ## FS converter
        mp4_fs_aac = self.print_acodec_from_mp4([vpath], only_sr=True)
        resampler = es.Resample(inputSampleRate=mp4_fs_aac, outputSampleRate=self.model_config['audio_fs_input'])
        
        if debug: tt1 = time.time()
        
        audio_es = mp.AudioFileClip(filename=vpath, fps=mp4_fs_aac)
        p_es_array = np.empty((len(frame_intervals),), dtype=object)

        print(vpath , frame_intervals)
        for i, (sf, ef, label) in enumerate(frame_intervals):
            st = sf / 24;et = ef / 24
            audio_es_cut = audio_es.subclip(t_start=st, t_end=et)
            aud_arr = audio_es_cut.to_soundarray()
            aud_arr_mono_single = np.mean(aud_arr, axis=1).astype(np.float32)
            aud_arr_essentia = resampler(aud_arr_mono_single)
            print(f"{sf} {ef} , {st:0.2f} {et:0.2f}")
                
            ## predict
            p_es = self.model(aud_arr_essentia)
            print("p_es",np.shape(p_es))
            p_es_array[i] = p_es

        audio_es.close()
        
        if debug: 
            tt2 = time.time()
            print(  "\n\n************** START sinet.get_sigmoid ***************",\
                    "\n\n",os.path.basename(vpath)," w/ ", mp4_fs_aac ,"hZ",\
                    "\n ",frame_intervals)
            
            for j in range(len(p_es_array)):
                #print("\nframe_interval",frame_intervals[j])
                print("interval",j,"@",np.shape(p_es_array[j]),frame_intervals[j][2])
                
                for i in range(len(self.model_config['anom_labels_i2'])):
                    label_i = self.model_config['anom_labels_i2'][i]
                    print("\t",label_i, self.model_config['anom_labels2'][i],"@max",np.amax(p_es_array[j][:, label_i]),"AVG",np.mean(p_es_array[j][:, label_i], axis=0))
                    
            print('\n\n\t p_es_array @ sinet', np.shape(p_es_array),'@',str(tt2 - tt1),"secs") #"expls", np.amax(p_es[:, 72])
            print("\n***************** END sinet.get_sigmoid ********************")
        else:
             for j in range(len(p_es_array)):
                print("interval",j,"@",np.shape(p_es_array[j]),frame_intervals[j][2])   
                 
        return p_es_array