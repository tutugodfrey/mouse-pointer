import os
import cv2
from numpy import ndarray
import logging

log = logging.getLogger()

class InputFeeder:
    def __init__(self, args):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''
        self.input_file = args.input
        

    def load_data(self):

        video_exts = ['.mp4']
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp',]
        if self.input_file == 'cam':
            self.input_type = 'cam'
        elif os.path.splitext(self.input_file)[-1] in video_exts:
            self.input_type = 'video'
        elif os.path.splitext(self.input_file)[-1] in image_exts:
            self.input_type = 'image'
        else:
            log.error('Unsupported file type. Please pass a video or image file or cam for camera')
            return

        if self.input_type=='video' or self.input_type == 'image':
            self.cap=cv2.VideoCapture(self.input_file)
            width, height = int(self.cap.get(3)), int(self.cap.get(4))
        elif self.input_type=='cam':
            self.cap=cv2.VideoCapture(0)
            width, height = int(self.cap.get(3)), int(self.cap.get(4))

        return width, height

    # def next_batch(self):
    #     '''
    #     Returns the next image from either a video file or webcam.
    #     If input_type is 'image', then it returns the same image.
    #     '''
    #     if self.input_type == 'image':
    #         yield self.cap
    #     else:
    #         while True:
    #             for _ in range(5):
    #                 flag, frame=self.cap.read()
    #             yield frame


    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type=='image':
            self.cap.release()

