import os
import sys
import cv2
import numpy as np
from argparse import ArgumentParser
import time
import logging

from head_pose_estimation import HeadPoseEstimation
from face_detection import FaceDetector
from facial_landmarks_detection import FacialLandmarksDetector
from gaze_estimation import GazeEstimation, main as gazer

from utils import OutputHandler
from input_feeder import InputFeeder
from mouse_controller import MouseController
from run_inference import run_inference

logging.getLogger().setLevel(logging.INFO)
parser = ArgumentParser()
CPU_EXTENSION_MAC = '/opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib'
parser.add_argument('-i', '--input', help='Path to input file or "cam" for camera stream', default='bin/demo.mp4')
parser.add_argument('-d', '--device', help='Hardware device to use for running inference', default='CPU')
parser.add_argument('-x', '--extensions', help='Path to CPU extensions for unsupported layers', default=CPU_EXTENSION_MAC)
parser.add_argument('-t', '--threshold', help='threshold for detecting faces with the face detection model', default=0.6)
parser.add_argument('-v', '--visualize', type=bool,  help='Control whether output of all intermediate models should be displayed. Enable visually all type of outputs. default 0. set to 1 if yes', default=0)
parser.add_argument('-vf', '--visualize_face', type=bool,  help='Control whether output of face detector model should be displayed default 0. set to 1 if yes', default=0)
parser.add_argument('-vl', '--visualize_landmarks', type=bool,  help='Control whether output of landmark model should be displayed default 0. set to 1 if yes', default=0)
parser.add_argument('-vh', '--visualize_head_pose', type=bool,  help='Control whether output of intermediate models should be displayed  for head pose. Default 0. set to 1 if yes', default=0)
parser.add_argument('-vg', '--visualize_gaze', type=bool,  help='Control whether output of of gaze model. Default 0. set to 1 if yes', default=0)
parser.add_argument('-vo', '--visualize_output', type=bool,  help='Control whether output of intermediate models should be displayed default 0. set to 1 if yes', default=0)
parser.add_argument('-fd', '--face_detection_precision',  help='Path to CPU extensions for unsupported layers', default='INT1')
parser.add_argument('-hd', '--head_pose_precision',  help='Path to CPU extensions for unsupported layers', default='FP16')
parser.add_argument('-gz', '--gaze_estimation_precision',  help='Path to CPU extensions for unsupported layers', default='FP16')
parser.add_argument('-lm', '--landmarks_precision',  help='Path to CPU extensions for unsupported layers', default='FP16')
parser.add_argument('-p', '--precision',  help='Control how much the mouse moves. Accepted values are "high", "low", "medium"', default='low')
parser.add_argument('-s', '--speed',  help='Control the speed of mouse move. Accepted values are "fast", "slow", "medium"', default='fast')


def main():
    # Get command line arguments
    args = parser.parse_args()
    device = args.device
    cpu_extensions = args.extensions
    threshold = args.threshold
    gaze_estimation_precision = args.gaze_estimation_precision
    head_pose_precision = args.head_pose_precision
    face_detection_precision = args.face_detection_precision
    landmarks_precision = args.landmarks_precision
    input_feeder = InputFeeder(args)
    control_mouse = MouseController(args)
    gaze_model = 'models/intel/gaze-estimation-adas-0002/{}/gaze-estimation-adas-0002'.format(gaze_estimation_precision)
    face_detector_model = 'models/intel/face-detection-adas-binary-0001/{}/face-detection-adas-binary-0001'.format(face_detection_precision)
    facial_landmark_model = 'models/intel/landmarks-regression-retail-0009/{}/landmarks-regression-retail-0009'.format(landmarks_precision)
    head_pose_model = 'models/intel/head-pose-estimation-adas-0001/{}/head-pose-estimation-adas-0001'.format(head_pose_precision)
    
    # Initialize the models
    face_detector = FaceDetector(face_detector_model, args)
    facial_landmarks = FacialLandmarksDetector(model_name=facial_landmark_model, device=device, extensions=cpu_extensions)
    head_pose_estimation = HeadPoseEstimation(model_name=head_pose_model, device=device, extensions=cpu_extensions)
    gaze_estimation = GazeEstimation(model_name=gaze_model, device=device, extensions=cpu_extensions)

    # Load the models
    start_time = time.time()
    face_detector.load_model()
    face_detector_loadtime = time.time() - start_time
    start_time = time.time()
    facial_landmarks.load_model()
    facial_landmark_loadtime = time.time() - start_time
    start_time = time.time()
    head_pose_estimation.load_model()
    head_pose_estimation_loadtime = time.time() - start_time
    start_time = time.time()
    gaze_estimation.load_model()
    gaze_estimation_loadtime = time.time() - start_time
    logging.info('FINISH LOADING MODELS')

    try:
        width, height = input_feeder.load_data()
    except TypeError:
        logging.error('Invalid file type.')
        return

    output_handler = OutputHandler(args)
    output_handler.initalize_video_writer(width, height)
    frame_count = 0
    start_time = 0
    capture = input_feeder.cap
    inputs = args.input
    if input_feeder.input_type == 'cam':
        inputs = 0
    else:
        capture.open(inputs)
    while capture.isOpened():
        flag, frame = capture.read()
    
        if start_time == 0:
            start_time = time.time()

        if inputs == 0 and time.time() - start_time >= 1:
            gaze_estimate = run_inference(frame, face_detector, facial_landmarks, head_pose_estimation, gaze_estimation, output_handler)
            if gaze_estimate is None:
                break

            if gaze_estimate[0][0]:
                x, y = gaze_estimate[0][:2]
                control_mouse.move(x, y)
            start_time = 0
            frame_count += 1
        elif not inputs == 0:
            gaze_estimate = run_inference(frame, face_detector, facial_landmarks, head_pose_estimation, gaze_estimation, output_handler)
            if gaze_estimate is None:
                break

            if gaze_estimate[0][0] and time.time() - start_time >= 0.5:
                x, y = gaze_estimate[0][:2]
                control_mouse.move(x, y)
                start_time = 0
            frame_count += 1

    input_feeder.close()
    logging.info('TOTOAL FRAMES PROCESSED: {}'.format(frame_count))
    logging.info('Time to load face detector model is {:.5f}'.format(face_detector_loadtime))
    logging.info('Time to load head pose estimation model is {:.5f}'.format(head_pose_estimation_loadtime))
    logging.info('Time to load facial landmarks model model is {:.5f}'.format(facial_landmark_loadtime))
    logging.info('Time to load gaze estimation model is {:.5f}'.format(gaze_estimation_loadtime))

if __name__ == '__main__':
    main()
