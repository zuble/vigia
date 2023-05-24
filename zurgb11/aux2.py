import cv2 , os

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions

# Load the Xception model with the top classification layer
xception_model = Xception(weights='imagenet')

# Load the video
video_path = '/raid/DATASETS/anomaly/XD_Violence/testing_copy/Skyfall.2012__#00-03-22_00-03-40_label_B6-0-0.mp4'
video = cv2.VideoCapture(video_path)


frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
random_frame_index = random.randint(0, frame_count - 1)
video.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)
ret, frame = video.read()

cv2.imshow('Random Frame', frame)
cv2.waitKey(4)
#cv2.destroyAllWindows()


## 1
#frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_resized = cv2.resize(frame, (299, 299))
frame_array = np.array(frame_resized)
frame_array = np.expand_dims(frame_array, axis=0)
frame_array_normalized = frame_array / 255.0

predictions_without_preprocess = xception_model.predict(frame_array_normalized)
print("Predictions without preprocess_input:")
print(decode_predictions(predictions_without_preprocess, top=3)[0])


## 2
frame_array_preprocessed = preprocess_input(frame_array)

predictions_with_preprocess = xception_model.predict(frame_array_preprocessed)
print("Predictions with preprocess_input:")
print(decode_predictions(predictions_with_preprocess, top=3)[0])


video.release()