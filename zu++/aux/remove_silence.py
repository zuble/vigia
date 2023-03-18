# From https://stackoverflow.com/questions/29547218/
# remove-silence-at-the-beginning-and-at-the-end-of-wave-files-with-pydub
from pydub import AudioSegment
import cv2

def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms
    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0  # ms
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size

    return trim_ms


if __name__ == '__main__':
    import sys, os

    in_folder = '/media/jtstudents/HDD/.zuble/xdviol/test/'
    in_format = "mp4"
    
    out_folder = '/media/jtstudents/HDD/.zuble/xdviol/test_audio/'
    out_format = "mp3"
    
    for root, dirs, files in os.walk(in_folder):
        for fn in files:
            if fn.find('.'+in_format) != -1:
                
                in_fp = os.path.join(root, fn)
                in_fn_wo_ext, in_ext = os.path.splitext(fn)
                out_fp = out_folder+in_fn_wo_ext+'.'+out_format
                
                print("CONVERTING",fn,"TO",out_format)
                AudioSegment.from_file(in_fp).export(out_fp, format=out_format)

                video = cv2.VideoCapture(in_fp)
                total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = video.get(cv2.CAP_PROP_FPS)
                video_time = total_frames/fps
                
                sound = AudioSegment.from_file(out_fp, format=out_format)
                
                #start_trim = detect_leading_silence(sound)
                #end_trim = detect_leading_silence(sound.reverse())
                #duration = len(sound)
                #trimmed_sound = sound[start_trim:duration-end_trim]
                #trimmed_sound.export(out_fp, format=out_format)
                
                # pydub does things in miliseconds
                video_time_ms = video_time * 1000
                print("\tNEW FILE HAS",video_time,
                "seconds\n-------------------------------------")
                trimmed_sound = sound[0:video_time_ms]
                trimmed_sound.export(out_fp, format=out_format)
