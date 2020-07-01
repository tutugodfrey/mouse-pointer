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
parser.add_argument('-v', '--visualize', type=bool,  help='Control whether output of all intermediate models should be displayed. Enable visually all type of outputs. default 0. set to 1 if yes', default=0)
parser.add_argument('-vf', '--visualize_face', type=bool,  help='Control whether output of face detector model should be displayed default 0. set to 1 if yes', default=0)
parser.add_argument('-vl', '--visualize_landmarks', type=bool,  help='Control whether output of landmark model should be displayed default 0. set to 1 if yes', default=0)
parser.add_argument('-vh', '--visualize_head_pose', type=bool,  help='Control whether output of intermediate models should be displayed  for head pose. Default 0. set to 1 if yes', default=0)
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
    inputs = args.input
    device = args.device
    cpu_extensions = args.extensions
    visualize = args.visualize
    visualize_face = args.visualize_face
    visualize_landmarks = args.visualize_landmarks
    visualize_head_pose = args.visualize_head_pose
    visualize_output = args.visualize_output
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

    # initialize video writers
    if visualize:
        cropped_face_video_writer = cv2.VideoWriter('cropped_face_video.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (width, height))
        output_video_with_drawings_writer = cv2.VideoWriter('output_video_with_drawings.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (width, height))
        landmark_drawings_writer = cv2.VideoWriter('landmarks_drawings.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (width, height))
        landmark_prediction_writer = cv2.VideoWriter('landmark_prediction.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (width, height))
        head_pose_angle_writer = cv2.VideoWriter('head_pose_angle_video.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (width, height))

    if visualize_face and not visualize:
        cropped_face_video_writer = cv2.VideoWriter('cropped_face_video.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (width, height))

    if visualize_output and not visualize:
        output_video_with_drawings_writer = cv2.VideoWriter('output_video_with_drawings.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (width, height))

    if visualize_landmarks and not visualize:
        landmark_drawings_writer = cv2.VideoWriter('landmarks_drawings.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (width, height))
        landmark_prediction_writer = cv2.VideoWriter('landmark_prediction.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (width, height))
    
    if visualize_head_pose and not visualize:
        head_pose_angle_writer = cv2.VideoWriter('head_pose_angle_video.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (width, height))


    for frame in input_feeder.next_batch():
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
                original_frame = frame[:]
                pred, frame = face_detector.predict(frame)
                faces, face_coords = face_detector.preprocess_output(pred, frame, 0.6)
                face = faces[0]
                x_min, y_min, x_max, y_max = face_coords[0]

                head_pose, head_pose_frame = head_pose_estimation.predict(face)
                head_pose_frame = head_pose_frame.transpose((1, 2, 0))
        except IndexError:
            print('No more frame to read')
            input_feeder.close()
            return
        except TypeError:
            print('No more frame to read from stream')
            input_feeder.close()
            return
        
        landmarks, landmarks_face = facial_landmarks.predict(face)
        landmarks_coords = facial_landmarks.preprocess_output(landmarks)



        eyes = facial_landmarks.get_eyes(landmarks_coords, landmarks_face)
        left_eye_image = eyes['left_eye']
        right_eye_image = eyes['right_eye']
        frame = frame.transpose((1, 2, 0))
        copied_frame = frame.copy()
        face = face.copy()

        # control all forms of visualizing output
        if visualize:
            landmark_predictions = output_handler.draw_facial_landmarks(landmarks_coords, landmarks_face, frame.shape[1], frame.shape[0])
            output_handler.write_frame(landmark_predictions, landmark_prediction_writer, width, height)

            # draw facial landmarks
            for key, coord in landmarks_coords.items():
                landmarks_frame = output_handler.draw_landmark(coord, pred[0][0], copied_frame, face, landmarks_face)
                output_handler.write_frame(landmarks_frame, landmark_drawings_writer, width, height)

            # draw bounding boxes on face and eyes
            # also draw circle on eyes
            frame = output_handler.draw_boxes(pred[0][0], frame, 0.6)
            frame = output_handler.draw_landmark(landmarks_coords['left_eye'], pred[0][0], frame, face, landmarks_face)
            frame = output_handler.draw_landmark(landmarks_coords['right_eye'], pred[0][0], frame, face, landmarks_face)
            output_handler.write_frame(face, cropped_face_video_writer, width, height)
            output_handler.write_frame(frame, output_video_with_drawings_writer, width, height)
            cv2.imwrite('left_eye_image.jpg', left_eye_image)
            cv2.imwrite('right_eye_image.jpg', right_eye_image)
        if visualize_face and not visualize:
            output_handler.write_frame(face, cropped_face_video_writer, width, height)

        if visualize_landmarks and not visualize:
            landmark_predictions = output_handler.draw_facial_landmarks(landmarks_coords, landmarks_face, frame.shape[1], frame.shape[0])
            output_handler.write_frame(landmark_predictions, landmark_prediction_writer, width, height)

            # draw facial landmarks
            for key, coord in landmarks_coords.items():
                landmarks_frame = output_handler.draw_landmark(coord, pred[0][0], copied_frame, face, landmarks_face)
                output_handler.write_frame(landmarks_frame, landmark_drawings_writer, width, height)

        if visualize_output and not visualize:
            frame = output_handler.draw_boxes(pred[0][0], frame, 0.6)
            frame = output_handler.draw_landmark(landmarks_coords['left_eye'], pred[0][0], frame, face, landmarks_face)
            frame = output_handler.draw_landmark(landmarks_coords['right_eye'], pred[0][0], frame, face, landmarks_face)
            output_handler.write_frame(frame, output_video_with_drawings_writer, width, height)
        
        if visualize_head_pose:
            head_frame = output_handler.draw_head_pose(head_pose, head_pose_frame, face, face_coords[0], frame)
            output_handler.write_frame(head_frame, head_pose_angle_writer, width, height)

        yaw = head_pose['angle_y_fc']
        pitch = head_pose['angle_p_fc']
        roll = head_pose['angle_r_fc']
        head_pose_angles = np.array([[yaw, pitch, roll]])
        gaze_estimate = gaze_estimation.predict({
            'left_eye_image': left_eye_image,
            'right_eye_image': right_eye_image,
            'head_pose_angles': head_pose_angles})
                
        # control_mouse = MouseController(precision, speed)
        # if gaze_estimate[0][0]:
        #     control_mouse.move(gaze_estimate[0][0], gaze_estimate[0][1])


if __name__ == '__main__':
    main()
