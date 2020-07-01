import cv2
import numpy as np
import math
class OutputHandler:
    def __init__(self):
        return None

    def draw_boxes(self, pred, image, threshold):
        '''
        Draw bounding boxes on image base on prediction (pred)
        pred: output from a object detection model [x, 7]
        image: original image to draw on
        threshold: confidence level to consider for drawing boxes
        return image with bounding boxes 
        '''

        coords = self.get_box_coordinates(pred, image, threshold)
        for coord in coords:
            x_min, y_min, x_max, y_max = coord
            image = cv2.rectangle(image,
                (x_min, y_min), 
                (x_max, y_max),
                color=(0, 225, 225),
                thickness=2)

        return image

    def draw_landmark(self, pred, face_pred, image, face, landmark_face):
        '''
        Draw bounding boxes and circle on image base on prediction (pred)
        pred: predicted coordinate of a facial landmark
        face_pred: prediction from the face_detection model (coordinate of detected faces as 2D array)
        image: original image that has been resized by face_detection model
        face: the crop of the face detected by face_detection model
        landmark_face: face resized by landmark_detection model 48x48
        return image with width drawings
        '''
        landmark_face = landmark_face.transpose((1,2,0))
        
        image_y = image.shape[0]
        image_x = image.shape[1]

        face_y = face.shape[0]
        face_x = face.shape[1]

        landmark_face_y = landmark_face.shape[0]
        landmark_face_x = landmark_face.shape[1]

        resize_ratio_y = landmark_face_y / face_y
        resize_ratio_x = landmark_face_x / face_x

        eye_pos_x = int((pred[0] * landmark_face_x) / resize_ratio_x)
        eye_pos_y = int((pred[1] * landmark_face_y) / resize_ratio_y)

        coords = self.get_box_coordinates(face_pred, image, 0.6)
        for coord in coords:
            x_min, y_min, x_max, y_max = coord

            diff_crop_x_min = x_min
            diff_crop_y_min = y_min
            diff_crop_x_max = image_x - x_max
            diff_crop_y_max = image_y - y_max

            eye_pos_x = eye_pos_x + diff_crop_x_min
            eye_pos_y = eye_pos_y + diff_crop_y_min
            image = cv2.circle(image,
                (eye_pos_x, eye_pos_y),
                radius=2,
                color=(225, 0, 225),
                thickness=1)

            image = cv2.rectangle(image,
                (eye_pos_x - 15, eye_pos_y - 15),
                (eye_pos_x + 15, eye_pos_y + 15),
                color=(0, 0, 225),
                thickness=1)

        return image

    def draw_facial_landmarks(self, landmarks_coords, frame, width, height):
        '''
        draw rectangle on the same image that was used
        for making prediction with facial landmark model
        landmarks_coords: key/value pair of the landmarks and their prediction
        frame: image used by landmark model for prediction, shape, (3, 48, 48)
        width: desired width of the image to return
        height: desired height of image to return
        return: frame with drawings
        '''
        frame = cv2.resize(frame.transpose((1, 2, 0)), (width, height))
        for key, coords in landmarks_coords.items():
            x_coord = coords[0]
            y_coord = coords[1]
            frame = cv2.circle(frame, (x_coord*width, y_coord*height), radius=2, color=(150, 100, 225), thickness=3)
            frame = cv2.rectangle(frame,
                (x_coord*width - 20, y_coord*height - 20),
                (x_coord*width + 20, y_coord*height + 20),
                color=(60, 80, 225), thickness=3)

        return frame
    
    def crop_image(self, coord, image):
        '''
        coords: array of coordinates x_min, y_min, x_max, y_max
        image: image to be cropped
        returns: cropped image
        '''
        x_min, y_min, x_max, y_max = coord
        image = image[y_min:y_max, x_min:x_max]

        return image

    def write_frame(self, frame, writer, width, height):
        '''
        frame: frame to be write as part of a video stream  shape (H x W x C)
        writer: An instance of cv2 Video writer
        width: desired width of the output video
        height: desired height of output video
        Note: width and height should be the same as set in the writer object.
        '''

        frame = cv2.resize(frame, (width, height))
        writer.write(frame)
    
    def get_box_coordinates(self, pred, image, threshold):
        '''
        pred: prediction from an face detection/ object detection model shape ([X x 7])
        typically [200x7] with the outer dimensions striped out
        image: image used for the predictions
        threshold: confidence threshold of boxes to be returned
        returns: an array of the coordinate of bounding boxes
        '''
        height = image.shape[0]
        width = image.shape[1]
        coords = []
        for box in pred:
            if box[2] >= threshold:
                x_min = int(box[3] * width)
                y_min = int(box[4] * height)
                x_max = int(box[5] * width)
                y_max = int(box[6] * height)
                coords.append([x_min, y_min, x_max, y_max])

        return coords

    def build_camera_matrix(self, center_of_face, focal_length):
        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        camera_matrix = np.zeros((3, 3), dtype='float32')
        camera_matrix[0][0] = focal_length
        camera_matrix[0][2] = cx
        camera_matrix[1][1] = focal_length
        camera_matrix[1][2] = cy
        camera_matrix[2][2] = 1

        return camera_matrix


    # source: udacity knowledge answer( https://knowledge.udacity.com/questions/171017)
    # source: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    def draw_axes(self, frame, center_of_face, yaw, pitch, roll, scale, focal_length):
        
        yaw *= np.pi / 180.0
        pitch *= np.pi / 180.0
        roll *= np.pi / 180.0
        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        Rx = np.array([[1, 0, 0],
                        [0, math.cos(pitch), -math.sin(pitch)],
                        [0, math.sin(pitch), math.cos(pitch)]])
        
        Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                        [0, 1, 0],
                        [math.sin(yaw), 0, math.cos(yaw)]])
        Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                        [math.sin(roll), math.cos(roll), 0],
                        [0, 0, 1]])
        R = Rz @ Ry @ Rx
        camera_matrix = self.build_camera_matrix(center_of_face, focal_length)
        xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
        yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
        zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
        zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)

        o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
        o[2] = camera_matrix[0][0]

        xaxis = np.dot(R, xaxis) + o
        yaxis = np.dot(R, yaxis) + o
        zaxis = np.dot(R, zaxis) + o
        zaxis1 = np.dot(R, zaxis1) + o
        xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
        # p2 = (int(xp2) + 40, int(yp2) + 40)
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 0, 225), 2)

        xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 225, 0), 2)

        xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
        yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
        p1 = (int(xp1), int(yp1))
        xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))

        cv2.line(frame, p1, p2, (225, 0, 0), 2)
        cv2.circle(frame, p2, 3, (225, 0, 0), 2)

        return frame
    
    def draw_head_pose(self, head_pose, head_pose_frame, face, face_coords, frame):
        '''
        Draw head pose angle on the main frame
        head_pose: output from the head pose model
        head_pose_frame: the resize face model used for estimating the head pose
        face: cropped frame from the face detection model
        face_coords: The coords of the bounding box for face base on the predictin
        frame the main frame use passed to the face detection model. shape (H x W X C)
        '''
        yaw = head_pose['angle_y_fc']
        pitch = head_pose['angle_p_fc']
        roll = head_pose['angle_r_fc']
        focal_length = 950.0
        scale = 50

        x_min, y_min, x_man, y_max = face_coords

        head_pose_y = head_pose_frame.shape[0]
        head_pose_x = head_pose_frame.shape[1]
        center_of_face_x = head_pose_x / 2
        center_of_face_y = head_pose_y / 2

        face_resize_ratio_x = head_pose_x / face.shape[1]
        face_resize_ratio_y = head_pose_y / face.shape[0]
        center_of_face_x = (center_of_face_x  / face_resize_ratio_x) + x_min
        center_of_face_y = (center_of_face_y  / face_resize_ratio_y) + y_min
        center_of_face = (center_of_face_x, center_of_face_y, 0)
        head_frame = self.draw_axes(frame, center_of_face, yaw, pitch, roll, scale, focal_length)
        
        return head_frame