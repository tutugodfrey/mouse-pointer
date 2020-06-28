import cv2

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
