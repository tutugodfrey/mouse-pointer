import os
import sys
import cv2
import numpy as np

from head_pose_estimation import HeadPoseEstimation
from face_detection import FaceDetector
from facial_landmarks_detection import FacialLandmarksDetector
from gaze_estimation import GazeEstimation, main as gazer

from draw_image import draw_boxes
from input_feeder import InputFeeder
from mouse_controller import MouseController

def main():
    CPU_EXTENSION_MAC = '/opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib'
    gaze_model = 'models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002'
    face_detector_model = 'models/intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001'
    facial_landmark_model = 'models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009'
    head_pose_model = 'models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001'

    image = 'bin/test-image1.jpg'

    input_feeder = InputFeeder('video', 'bin/demo.mp4')
    # input_feeder = InputFeeder('cam', 'bin/demo.mp4')
    input_feeder.load_data()

    for image in input_feeder.next_batch():
        # Initialize the models
        face_detector = FaceDetector(model_name=face_detector_model, device='CPU', extensions=CPU_EXTENSION_MAC)
        facial_landmarks = FacialLandmarksDetector(model_name=facial_landmark_model, device='CPU', extensions=CPU_EXTENSION_MAC)
        head_pose_estimation = HeadPoseEstimation(model_name=head_pose_model, device='CPU', extensions=CPU_EXTENSION_MAC)
        gaze_estimation = GazeEstimation(model_name=gaze_model, device='CPU', extensions=CPU_EXTENSION_MAC)

        # Load the models
        face_detector.load_model()
        facial_landmarks.load_model()
        head_pose_estimation.load_model()
        gaze_estimation.load_model()

        try:
            if image is None:
                raise TypeError
            else:
                pred = face_detector.predict(image)
                image = face_detector.preprocess_output(pred, image)

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
        cv2.imwrite('new_left_eyes.jpg', left_eye_image)
        cv2.imwrite('new_right_eyes.jpg', right_eye_image)

        gaze_estimate = gaze_estimation.predict({
            'left_eye_image': eyes['left_eye'],
            'right_eye_image': eyes['right_eye'],
            'head_pose_angles': head_pose})
        
        control_mouse = MouseController('low', 'fast')
        if gaze_estimate[0][0]:
            control_mouse.move(gaze_estimate[0][0], gaze_estimate[0][1])

if __name__ == '__main__':
    main()