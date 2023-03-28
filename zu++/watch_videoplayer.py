import cv2
import threading
import queue

class CircularBuffer:
    def __init__(self, size):
        self.buffer = [None] * size
        self.size = size
        self.start = 0
        self.end = 0
        self.count = 0
        self.lock = threading.Lock()
        self.not_full = threading.Condition(self.lock)
        self.not_empty = threading.Condition(self.lock)

    def put(self, item):
        with self.not_full:
            while self.count == self.size:
                self.not_full.wait()
            self.buffer[self.end] = item
            self.end = (self.end + 1) % self.size
            self.count += 1
            self.not_empty.notify()

    def get(self):
        with self.not_empty:
            while self.count == 0:
                self.not_empty.wait()
            item = self.buffer[self.start]
            self.start = (self.start + 1) % self.size
            self.count -= 1
            self.not_full.notify()
            return item

class VideoPlayer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.frame_buffer = CircularBuffer(100)
        self.quit_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()

    def start(self):
        self.video_thread = threading.Thread(target=self.read_frames)
        self.video_thread.start()

    def read_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        while not self.quit_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_buffer.put(frame)
        cap.release()

    def play(self):
        while not self.quit_event.is_set():
            frame = self.frame_buffer.get()
            cv2.imshow('Video', frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                self.quit_event.set()
            elif key == ord(' '):
                self.pause_event.clear()
                cv2.waitKey()
                self.pause_event.set()
            elif key == ord('p'):
                self.pause_event.set()
                while self.pause_event.is_set():
                    cv2.waitKey(1)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = "/raid/DATASETS/anomaly/XD_Violence/testing/v=wQrV75N2BrI__#1_label_A.mp4"
    player = VideoPlayer(video_path)
    player.start()
    player.play()