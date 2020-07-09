import cv2
import numpy as np
import math
class OutputHandler:
    def __init__(self, args):
        self.visualize = args.visualize
        self.visualize_face = args.visualize_face
        self.visualize_landmarks = args.visualize_landmarks
        self.visualize_head_pose = args.visualize_head_pose
        self.visualize_gaze = args.visualize_gaze
        self.visualize_output = args.visualize_output
        self.threshold = args.threshold

        # set frames
        self.frame = None
        self.face_frame = None
        self.head_pose_frame = None
        self.landmarks_frame = None
        self.landmarks_coords = None
        self.gaze_frame = None
        # self.landmarks_face = None
        self.head_pose_pred = None
        self.face_coords = None
        self.gaze_estimate = None
        self.right_eye_center = None
        self.left_eye_center = None
        self.center_of_face = None
        
    def initalize_video_writer(self, width=None, height=None):
        self.width, self.height = width, height

        # initialize video writers
        if self.visualize:
            self.cropped_face_video_writer = cv2.VideoWriter('cropped_face_video.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (width, height))
            self.output_video_with_drawings_writer = cv2.VideoWriter('output_video_with_drawings.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (width, height))
            self.landmark_drawings_writer = cv2.VideoWriter('landmarks_drawings.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (width, height))
            self.landmark_prediction_writer = cv2.VideoWriter('landmark_prediction.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (width, height))
            self.head_pose_angle_writer = cv2.VideoWriter('head_pose_angle_video.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (width, height))


        if self.visualize_face and not self.visualize:
            self.cropped_face_video_writer = cv2.VideoWriter('cropped_face_video.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (width, height))

        if self.visualize_output and not self.visualize:
            self.output_video_with_drawings_writer = cv2.VideoWriter('output_video_with_drawings.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (width, height))

        if self.visualize_landmarks and not self.visualize:
            self.landmark_drawings_writer = cv2.VideoWriter('landmarks_drawings.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (width, height))
            self.landmark_prediction_writer = cv2.VideoWriter('landmark_prediction.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (width, height))
        
        if self.visualize_head_pose and not self.visualize:
            self.head_pose_angle_writer = cv2.VideoWriter('head_pose_angle_video.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (width, height))

        if self.visualize_gaze:
            self.gaze_direction_writer = cv2.VideoWriter('gaze_direction_video.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (width, height))
    
    def prepare_visualizer(self):
        self.face_coords = self.get_box_coordinates(self.face_pred, self.frame, 0.6)
        self.right_eye_center = self.get_point_of_landmark(self.landmarks_coords['right_eye'], self.face_coords, self.frame, self.face_frame, self.landmarks_frame)
        self.left_eye_center = self.get_point_of_landmark(self.landmarks_coords['left_eye'], self.face_coords, self.frame, self.face_frame, self.landmarks_frame)

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

    def draw_landmark(self, image, point):
        '''
        Draw bounding boxes and circle on image base on prediction (pred)
        landmark_pred: predicted coordinate of a facial landmark
        face_pred: prediction from the face_detection model (coordinate of detected faces as 2D array)
        image: original image that has been resized by face_detection model
        face: the crop of the face detected by face_detection model
        landmark_face: face resized by landmark_detection model 48x48
        return image with width drawings
        '''
        landmark_point_x, landmark_point_y = point
        image = cv2.circle(image,
            (landmark_point_x, landmark_point_y),
            radius=2,
            color=(225, 0, 225),
            thickness=1)

        image = cv2.rectangle(image,
            (landmark_point_x - 15, landmark_point_y - 15),
            (landmark_point_x + 15, landmark_point_y + 15),
            color=(0, 0, 225),
            thickness=1)

        return image
        
    def get_point_of_landmark(self, landmark_pred, face_coords, image, face, landmark_frame):
        '''
        Get the point of predicted landmark with reference to the original image
        landmark_pred: predicted coordinate of a facial landmark
        face_pred: prediction from the face_detection model (coordinate of detected faces as 2D array)
        image: original image that has been resized by face_detection model
        face: the crop of the face detected by face_detection model
        landmark_frame: face resized by landmark_detection model 48x48
        return image with width drawings
        '''
        landmark_frame = landmark_frame.transpose((1,2,0))
        
        image_y = image.shape[0]
        image_x = image.shape[1]

        face_y = face.shape[0]
        face_x = face.shape[1]

        landmark_frame_y = landmark_frame.shape[0]
        landmark_frame_x = landmark_frame.shape[1]

        resize_ratio_y = landmark_frame_y / face_y
        resize_ratio_x = landmark_frame_x / face_x

        landmark_pos_x = int((landmark_pred[0] * landmark_frame_x) / resize_ratio_x)
        landmark_pos_y = int((landmark_pred[1] * landmark_frame_y) / resize_ratio_y)

        for coord in face_coords:
            x_min, y_min, x_max, y_max = coord

            diff_crop_x_min = x_min
            diff_crop_y_min = y_min
            diff_crop_x_max = image_x - x_max
            diff_crop_y_max = image_y - y_max

            landmark_pos_x = landmark_pos_x + diff_crop_x_min
            landmark_pos_y = landmark_pos_y + diff_crop_y_min

        return landmark_pos_x, landmark_pos_y

    def get_center_of_face(self):
        '''
        Calculate the center of the detected face
        Set the center_of_face class variable
        return the center of face
        '''
        x_min, y_min, x_man, y_max = self.face_coords[0]
        head_pose_y = self.head_pose_frame.shape[0]
        head_pose_x = self.head_pose_frame.shape[1]
        center_of_face_x = head_pose_x / 2
        center_of_face_y = head_pose_y / 2

        face_resize_ratio_x = head_pose_x / self.face_frame.shape[1]
        face_resize_ratio_y = head_pose_y / self.face_frame.shape[0]
        center_of_face_x = (center_of_face_x  / face_resize_ratio_x) + x_min
        center_of_face_y = (center_of_face_y  / face_resize_ratio_y) + y_min
        self.center_of_face = (center_of_face_x, center_of_face_y, 0)
            
        return self.center_of_face

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
        image: image used for the predictions, structure (H x W x C)
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
        p2 = (int(xp2) + 40, int(yp2) + 40)
        # p2 = (int(xp2), int(yp2))
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

        return self.frame
    
    def draw_head_pose(self):
        '''
        Draw head pose angle on the main frame
        head_pose: output from the head pose model
        head_pose_frame: the resize face model used for estimating the head pose
        face: cropped frame from the face detection model
        face_coords: The coords of the bounding box for face base on the predictin
        frame the main frame use passed to the face detection model. shape (H x W X C)
        '''
        yaw = self.head_pose_pred['angle_y_fc']
        pitch = self.head_pose_pred['angle_p_fc']
        roll = self.head_pose_pred['angle_r_fc']
        focal_length = 950.0
        scale = 50
        center_of_face = self.get_center_of_face()
        head_frame = self.draw_axes(self.frame, center_of_face, yaw, pitch, roll, scale, focal_length)
        
        return head_frame
    
    def output_visualizer(self):
        # control all forms of visualizing output
        if self.visualize:
            frame = self.frame.copy()
            landmark_predictions = self.draw_facial_landmarks(self.landmarks_coords, self.landmarks_frame, frame.shape[1], frame.shape[0])
            self.write_frame(landmark_predictions, self.landmark_prediction_writer, self.width, self.height)

            # draw facial landmarks
            for key, coord in landmarks_coords.items():
                landmarks_frame = self.draw_landmark(coord, pred[0][0], copied_frame, self.face_frame, self.landmarks_frame)
                self.write_frame(landmarks_frame, self.landmark_drawings_writer,self. width, self.height)

            # draw bounding boxes on face and eyes
            # also draw circle on eyes
            frame = self.draw_boxes(self.face_pred, frame, 0.6)
            frame = self.draw_landmark(frame, self.left_eye_center)
            frame = self.draw_landmark(frame, self.right_eye_center)
            self.write_frame(_frameface, self.cropped_face_video_writer, self.width, self.height)
            self.write_frame(frame, self.output_video_with_drawings_writer, self.width, self.height)

        if self.visualize_face and not self.visualize:
            self.write_frame(face, self.cropped_face_video_writer, self.width, self.height)

        if self.visualize_landmarks and not self.visualize:
            frame = self.frame.copy()
            landmark_predictions = self.draw_facial_landmarks(self.landmarks_coords, self.landmarks_frame, frame.shape[1], frame.shape[0])
            self.write_frame(landmark_predictions, self.landmark_prediction_writer, self.width, self.height)

            # draw facial landmarks
            for key, coord in self.landmarks_coords.items():
                landmark_point = self.get_point_of_landmark(coord, self.face_coords, frame, self.face_frame, self.landmarks_frame)
                landmarks_frame = self.draw_landmark(frame, landmark_point)
                self.write_frame(landmarks_frame, self.landmark_drawings_writer, self.width, self.height)

        if self.visualize_output and not self.visualize:
            frame = self.frame.copy()
            frame = self.draw_boxes(self.face_pred, frame, 0.6)
            frame = self.draw_landmark(frame, self.left_eye_center)
            frame = self.draw_landmark(frame, self.right_eye_center)
            self.write_frame(frame, self.output_video_with_drawings_writer, self.width, self.height)
        
        if self.visualize_head_pose:
            head_frame = self.draw_head_pose()
            self.write_frame(head_frame, self.head_pose_angle_writer, self.width, self.height)
        
        if self.visualize_gaze:
            frame = self.frame.copy()
            x, y = self.gaze_estimate[0, :2]
            head_frame = self.draw_head_pose()
            frame = cv2.arrowedLine(frame, self.left_eye_center, (int(self.left_eye_center[0]+x*200), int(self.left_eye_center[1]-y*200)), (0, 120, 20), 2)
            frame = cv2.arrowedLine(frame, self.right_eye_center, (int(self.right_eye_center[0]+x*200), int(self.right_eye_center[1]-y*200)), (0, 120, 20), 2)

            self.write_frame(frame, self.gaze_direction_writer, self.width, self.height)
            cv2.imwrite('gaze.jpg', frame)
