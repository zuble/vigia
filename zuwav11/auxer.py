import cv2

video = cv2.VideoCapture('/raid/DATASETS/anomaly/XD_Violence/testing_copy/v=38GQ9L2meyE__#1_label_B6-0-0.mp4')
while video.isOpened() :
    sucess, frame = video.read()
    if sucess:
        cv2.imshow("w",frame)
        key = cv2.waitKey(1)
        
        if key == ord('q'): break  # quit
        if key == ord(' '):  # pause
            while True:
                key = cv2.waitKey(1)
                if key == ord(' '):break
    else: break
video.release()
cv2.destroyAllWindows()