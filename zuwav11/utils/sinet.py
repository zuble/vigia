import json , time, subprocess

import numpy as np

import moviepy.editor as mp
import essentia 
#print(essentia.__version__)
print(essentia.__file__)
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
        
        print("\n\nSINET VERSION", self.model_config["sinet_version"])
        
    
    def print_acodec_from_mp4(self, data, printt=False, only_sr=False):
        out = []
        for i in range(len(data)):
            output = subprocess.check_output('ffprobe -hide_banner -v error -select_streams a:0 -show_entries stream=codec_name,channels,sample_rate,bit_rate -of default=noprint_wrappers=1 -of compact=p=0:nk=1 ' + str('"' + data[i] + '"'), shell=True)
            output = str(output).replace("\\n", "").replace("b", "").replace("'", "").splitlines()[0]
            out.append(output)
            if printt: print(output)
        if only_sr: return int(str(out[0]).split('|')[1])
        else: return out

    def get_sigmoid(self, vpath , debug = False):

        ## FS converter
        mp4_fs_aac = self.print_acodec_from_mp4([vpath], only_sr=True)
        resampler = es.Resample(inputSampleRate=mp4_fs_aac, outputSampleRate=self.model_config['audio_fs_input'])
        
        if debug: 
            print("\n\n************** get_sigmoid ***************")
            print("mp4_fs_aac", mp4_fs_aac)
            tt1 = time.time()
            
        ## process aud data
        audio_es = mp.AudioFileClip(filename=vpath, fps=mp4_fs_aac)
        aud_arr = audio_es.to_soundarray()
        aud_arr_mono_single = np.mean(aud_arr, axis=1).astype(np.float32)
        aud_arr_essentia = resampler(aud_arr_mono_single)

        ## predict
        p_es = self.model(aud_arr_essentia)

        if debug: 
            tt2 = time.time()
            print(np.shape(p_es), '@', str(tt2 - tt1)) #"expls", np.amax(p_es[:, 72])
            print("\nMAX aas for anom labels 2")
            for i in range(len(self.model_config['anom_labels_i2'])):
                label_i = self.model_config['anom_labels_i2'][i]
                print(label_i, self.model_config['anom_labels2'][i], np.amax(np.asarray(p_es)[:, label_i]))
            print('\n\n\t p_es shape', p_es.shape)
            print("\n\n******************************************")
            
        if self.model_config['full_or_max'] == 'full': return p_es
        elif self.model_config['full_or_max'] == 'max': return np.max(p_es, axis=0)