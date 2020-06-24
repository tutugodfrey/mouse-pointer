import os
import sys
import cv2
import numpy as np
from argparse import ArgumentParser

from head_pose_estimation import HeadPoseEstimation
from face_detection import FaceDetector
from facial_landmarks_detection import FacialLandmarksDetector
from gaze_estimation import GazeEstimation, main as gazer

from utils import OutputHandler
from input_feeder import InputFeeder
from mouse_controller import MouseController

parser = ArgumentParser()
CPU_EXTENSION_MAC = '/opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib'
parser.add_argument('-i', '--input', help='Path to input file or "cam" for camera stream', default='bin/demo.mp4')
parser.add_argument('-d', '--device', help='Hardware device to use for running inference', default='CPU')
parser.add_argument('-x', '--extensions', help='Path to CPU extensions for unsupported layers', default=CPU_EXTENSION_MAC)
parser.add_argument('-v', '--visualize', type=bool,  help='Control whether output of intermediate models should be displayed default 0. set to 1 if yes', default=0)
parser.add_argument('-fd', '--face_detection_precision',  help='Path to CPU extensions for unsupported layers', default='INT1')
parser.add_argument('-hd', '--head_pose_precision',  help='Path to CPU extensions for unsupported layers', default='FP16')
parser.add_argument('-gz', '--gaze_estimation_precision',  help='Path to CPU extensions for unsupported layers', default='FP16')
parser.add_argument('-lm', '--landmarks_precision',  help='Path to CPU extensions for unsupported layers', default='FP16')
parser.add_argument('-p', '--precision',  help='Control how much the mouse moves. Accepted values are "high", "low", "medium"', default='low')
parser.add_argument('-s', '--speed',  help='Control the speed of mouse move. Accepted values are "fast", "slow", "medium"', default='fast')


def main():
    # Get command line arguments
    args = parser.parse_args()
    inputs = args.input
    device = args.device
    cpu_extensions = args.extensions
    visualize = args.visualize
    precision = args.precision
    speed = args.speed
    gaze_estimation_precision = args.gaze_estimation_precision
    head_pose_precision = args.head_pose_precision
    face_detection_precision = args.face_detection_precision
    landmarks_precision = args.landmarks_precision

    gaze_model = 'models/intel/gaze-estimation-adas-0002/{}/gaze-estimation-adas-0002'.format(gaze_estimation_precision)
    face_detector_model = 'models/intel/face-detection-adas-binary-0001/{}/face-detection-adas-binary-0001'.format(face_detection_precision)
    facial_landmark_model = 'models/intel/landmarks-regression-retail-0009/{}/landmarks-regression-retail-0009'.format(landmarks_precision)
    head_pose_model = 'models/intel/head-pose-estimation-adas-0001/{}/head-pose-estimation-adas-0001'.format(head_pose_precision)

    video_exts = ['.mp4']
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp',]
    if inputs == 'cam':
        input_feeder = InputFeeder(inputs)
    elif os.path.splitext(inputs)[-1] in video_exts:
        input_feeder = InputFeeder('video', inputs)
    elif os.path.splitext(inputs)[-1] in image_exts:
        input_feeder = InputFeeder('image', inputs)
    else:
        print('Unsupported file type. Please pass a video or image file or cam for camera')
        return
    



    
    width, height = input_feeder.load_data()
    if visualize:
        video_writer = cv2.VideoWriter('cropped_face_video.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (width, height))
        video_writer2 = cv2.VideoWriter('original_video_with_frame.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (width, height))

    for frame in input_feeder.next_batch():
        key = cv2.waitKey(100)

        # Initialize the models
        face_detector = FaceDetector(model_name=face_detector_model, device=device, extensions=cpu_extensions)
        facial_landmarks = FacialLandmarksDetector(model_name=facial_landmark_model, device=device, extensions=cpu_extensions)
        head_pose_estimation = HeadPoseEstimation(model_name=head_pose_model, device=device, extensions=cpu_extensions)
        gaze_estimation = GazeEstimation(model_name=gaze_model, device=device, extensions=cpu_extensions)

        # Load the models
        face_detector.load_model()
        facial_landmarks.load_model()
        head_pose_estimation.load_model()
        gaze_estimation.load_model()
        output_handler = OutputHandler()
        try:
            if frame is None:
                raise TypeError
            else:
                pred = face_detector.predict(frame)
                image = face_detector.preprocess_output(pred, frame, 0.6)
                frame = output_handler.draw_boxes(pred[0][0], frame, 0.6)
                if visualize:
                    output_handler.write_frame(image[0], video_writer, width, height)
                    output_handler.write_frame(frame, video_writer2, width, height)

            head_pose = head_pose_estimation.predict(image[0])
            head_pose = np.array([head_pose])
        except IndexError:
            print('No more frame to read')
            input_feeder.close()
            return
        except TypeError:
            print('No more frame to read from stream')
            input_feeder.close()
            return
        
        landmarks = facial_landmarks.predict(image[0])
        eyes_coords = facial_landmarks.preprocess_output(landmarks[0])
        eyes = facial_landmarks.get_eyes(eyes_coords, image[0])
        left_eye_image = eyes['left_eye']
        right_eye_image = eyes['right_eye']
        if visualize:
            cv2.imwrite('new_left_eyes.jpg', left_eye_image)
            cv2.imwrite('new_right_eyes.jpg', right_eye_image)

        gaze_estimate = gaze_estimation.predict({
            'left_eye_image': eyes['left_eye'],
            'right_eye_image': eyes['right_eye'],
            'head_pose_angles': head_pose})
        
        control_mouse = MouseController(precision, speed)
        if gaze_estimate[0][0]:
            control_mouse.move(gaze_estimate[0][0], gaze_estimate[0][1])

if __name__ == '__main__':
    main()
