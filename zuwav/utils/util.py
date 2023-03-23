import os, subprocess
from essentia.standard import *

SERVER_TEST_PATH = '/raid/DATASETS/anomaly/XD_Violence/testing'
SERVER_TEST_AUD_ORIG_PATH = '/raid/DATASETS/anomaly/XD_Violence/aud/testing/original'
SERVER_TEST_AUD_MONO_PATH = '/raid/DATASETS/anomaly/XD_Violence/aud/testing/mono'


def load_xdv_test(aac_path):
    mp4_paths, mp4_labels = [],[]
    for root, dirs, files in os.walk(SERVER_TEST_PATH):
        for file in files:
            if file.find('.mp4') != -1:
                mp4_paths.append(os.path.join(root, file))
    mp4_paths.sort()
    for i in range(len(mp4_paths)):            
        if 'label_A' in  file:mp4_labels.append(0)
        else:mp4_labels.append(1)
    
    print('acc_path',aac_path)
    aac_paths, aac_labels = [],[]                            
    for root, dirs, files in os.walk(aac_path):
        for file in files:
            if file.find('.aac') != -1:
                aac_paths.append(os.path.join(root, file))
    aac_paths.sort()
    for i in range(len(aac_paths)):               
        if 'label_A' in  file:aac_labels.append(0)
        else:aac_labels.append(1)                
    
    return mp4_paths,mp4_labels,aac_paths,aac_labels


from pylab import plot, show, figure, imshow
#%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 6) # set plot sizes to something larger than default

def plot_mfcc_melbands(audio):
    plot(audio[1*44100:2*44100])
    plt.title("This is how the 2nd second of this audio looks like:")
    show()
    
    w = Windowing(type = 'hann')
    spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
    mfcc = MFCC()
    
    mfccs = []
    melbands = []
    melbands_log = []

    logNorm = UnaryOperator(type='log')
    
    for frame in FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        mfccs.append(mfcc_coeffs)
        melbands.append(mfcc_bands)
        melbands_log.append(logNorm(mfcc_bands))

    # transpose to have it in a better shape
    # we need to convert the list to an essentia.array first (== numpy.array of floats)
    mfccs = essentia.array(mfccs).T
    melbands = essentia.array(melbands).T
    melbands_log = essentia.array(melbands_log).T

    # and plot
    imshow(melbands[:,:], aspect = 'auto', origin='lower', interpolation='none')
    plt.title("Mel band spectral energies in frames")
    show()

    imshow(melbands_log[:,:], aspect = 'auto', origin='lower', interpolation='none')
    plt.title("Log-normalized mel band spectral energies in frames")
    show()

    imshow(mfccs[1:,:], aspect='auto', origin='lower', interpolation='none')
    plt.title("MFCCs in frames")
    show()



def print_acodec_from_mp4(data,printt=False):
    out=[]
    for i in range(len(data)):
        output = subprocess.check_output('ffprobe -hide_banner -v error -select_streams a:0 -show_entries stream=codec_name,channels,sample_rate,bit_rate -of default=noprint_wrappers=1 -of compact=p=0:nk=1 '+str('"'+data[i]+'"'), shell=True)
        output = str(output).replace("\\n","").replace("b","").replace("'","").splitlines()[0]
        out.append(output)
        if printt:print(output)
    #os.system('ffprobe -hide_banner -show_streams -select_streams a:0 -i '+str('"'+test_fn[1]+'"'))
    #os.system('ffprobe -hide_banner '+str('"'+test_fn[1]+'"'))
    return out

# convert mp4/multiple audio channels to mp4+mono https://superuser.com/questions/1711628/5-1-to-mono-using-ffmpeg
def conv_mp4_to_aac(mp4_paths,dest_path,channels,dry_run=True):
    print(dest_path)
    for path in mp4_paths:
        out_fn = os.path.join(dest_path,os.path.splitext(os.path.basename(path))[0]+'.aac')
        print("\nconverting ",os.path.basename(path),'\n\t\t\t',os.path.basename(out_fn))
        print("\n\nffmpeg -nostats -hide_banner -v warning -i "+str('"'+path+'"')+" -ac "+str(channels)+' '+str('"'+out_fn+'"'))
        if not dry_run:
            command = "ffmpeg -nostats -hide_banner -v warning -i "+str('"'+path+'"')+" -ac "+str(channels)+' '+str('"'+out_fn+'"')
            os.system(command)
