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

        height = image.shape[0]
        width = image.shape[1]
        for box in pred:
            if box[2] >= threshold:
                x_min = int(box[3] * width)
                y_min = int(box[4] * height)
                x_max = int(box[5] * width)
                y_max = int(box[6] * height)
                image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 0, 225), thickness=2)
        return image
    
    def crop_image(self, coords, image):
        '''
        coords: array of coordinates x_min, y_min, x_max, y_max
        image: image to be cropped
        returns: cropped image
        '''

        height = image.shape[0]
        width = image.shape[1]
        if len(coords) == 4:
            x_min = int(coords[0] * width)
            y_min = int(coords[1] * height)
            x_max = int(coords[2] * width)
            y_max = int(coords[3] * height)
            image = image[y_min:y_max, x_min:x_max]
            return image

    def write_frame(self, frame, writer, width, height):
        '''
        frame: frframe to be write as part of a video stream
        writer: An instance of cv2 Video writer
        width: desired width of the output video
        height: desired height of output video
        Note: width and height should be the same as set in the writer object.
        '''
        
        frame = cv2.resize(frame, (width, height))
        writer.write(frame)
