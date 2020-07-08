import cv2
import numpy as np
import logging
import time

# Track inference time for each model
# Will accumulate the inference time for processing frames
face_detector_inference_time = 0
facial_landmark_estimation_inference_time = 0
head_pose_estimation_inference_time = 0
gaze_estimation_inference_time = 0
frame_count = 0
def run_inference(frame, face_detector, facial_landmarks, head_pose_estimation, gaze_estimation, output_handler):
    global face_detector_inference_time
    global facial_landmark_estimation_inference_time
    global head_pose_estimation_inference_time
    global gaze_estimation_inference_time
    global frame_count

    face_frame = None
    try:
        if type(frame) is None:
            raise TypeError
        else:
            original_frame = frame[:]
            start_time = time.time()
            face_pred, frame = face_detector.predict(frame)
            inference_time = time.time() - start_time
            face_detector_inference_time  += inference_time
            frame_copied = frame[:]
            frame_copied = frame_copied.transpose((1, 2, 0))
            coords = output_handler.get_box_coordinates(face_pred[0][0], frame_copied, output_handler.threshold)
            faces = []
            for coord in coords:
                cropped_image = output_handler.crop_image(coord, frame_copied)
                faces.append(cropped_image)

            if faces and faces[0].shape[0] and faces[0].shape[1]:
                face_frame = faces[0]
    except IndexError:
        logging.info('No more frame to read')
        return
    except TypeError:
        logging.info('No more frame to read from stream')
        logging.info('time to run inference on in by Face detection model is {:.5f}'.format(face_detector_inference_time / frame_count))
        logging.info('time to run inference on in by head pose estimation model is {:.5f}'.format(head_pose_estimation_inference_time / frame_count))
        logging.info('time to run inference on in by facial landmarks model is {:.5f}'.format(facial_landmark_estimation_inference_time / frame_count))
        logging.info('time to run inference on in by gaze estimation model is {:.5f}'.format(gaze_estimation_inference_time / frame_count))
        logging.info('Total frames processed {}'.format(frame_count))
        return
    if face_frame is not None:
        start_time = time.time()
        landmarks, landmarks_frame = facial_landmarks.predict(face_frame)
        inference_time = time.time() - start_time
        facial_landmark_estimation_inference_time +=  inference_time

        landmarks_coords = facial_landmarks.preprocess_output(landmarks)
        start_time = time.time()
        head_pose_pred, head_pose_frame = head_pose_estimation.predict(face_frame)
        inference_time = time.time() - start_time
        head_pose_estimation_inference_time += inference_time

        eyes = facial_landmarks.get_eyes(landmarks_coords, landmarks_frame)
        left_eye_image = eyes['left_eye']
        right_eye_image = eyes['right_eye']
        yaw = head_pose_pred['angle_y_fc']
        pitch = head_pose_pred['angle_p_fc']
        roll = head_pose_pred['angle_r_fc']
        head_pose_angles = np.array([[yaw, pitch, roll]])

        start_time = time.time()
        gaze_estimate = gaze_estimation.predict({
            'left_eye_image': left_eye_image,
            'right_eye_image': right_eye_image,
            'head_pose_angles': head_pose_angles})
        inference_time = time.time() - start_time
        gaze_estimation_inference_time +=  inference_time
        frame_count += 1

        # Visualize output
        # Setting variables for visualizing the output
        output_handler.frame = frame.transpose((1, 2, 0))
        output_handler.face_frame = face_frame
        output_handler.head_pose_frame = head_pose_frame.transpose((1, 2, 0))
        output_handler.landmarks_frame = landmarks_frame
        output_handler.face_pred = face_pred[0][0]
        output_handler.landmarks_coords = landmarks_coords
        output_handler.head_pose_pred = head_pose_pred
        output_handler.gaze_estimate = gaze_estimate
        output_handler.prepare_visualizer()
        output_handler.output_visualizer()

        return gaze_estimate
