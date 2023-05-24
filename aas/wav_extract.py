from moviepy.editor import *

def convert_video_to_wav(video_path, output_wav_path, sampling_rate=16000):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.set_fps(sampling_rate).write_audiofile(output_wav_path)

fn = 'Bullet.in.the.Head.1990__#02-02-00_02-05-22_label_B2-B6-G'
video_path = "/raid/DATASETS/anomaly/XD_Violence/testing_copy/"+fn+'.mp4'
output_wav_path = "/raid/DATASETS/.zuble/vigia/zuwav11/aas/"+fn+".wav"
convert_video_to_wav(video_path, output_wav_path, sampling_rate=16000)