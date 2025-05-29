
import cv2


class VideoRecorder:
    def __init__(self, output_path, fps=2):
        self.output_path = output_path
        self.fps = fps
        self.video_writer = None

    def create_writter(self, width=None, height=None):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))

    def record_frame(self, frame):
        '''
        frame should be a numpy array of shape (height, width, 3) and dtype uint8
        '''
        if self.video_writer is None:
            self.create_writter()
        if self.video_writer is not None:
            self.video_writer.write(frame)

    def stop_recording(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def save_video(self):
        if self.video_writer is not None:
            self.stop_recording()
