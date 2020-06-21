'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import sys
import cv2
from openvino.inference_engine import IENetwork, IECore

class Model_X:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''

        self.model_name = model_name
        self.device = device
        self.extensions = extensions

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        model_bin = self.model_name + '.bin'
        model_xml = self.model_name + '.xml'
        self.model = IENetwork(model_xml, model_bin)
        self.plugin = IECore()
        if self.extensions:
            self.plugin.add_extension(self.extensions, self.device)
        self.exec_net = self.plugin.load_network(self.model, device_name=self.device, num_requests=1)
        self.check_model()


    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        frame = self.preprocess_input(image)
        self.exec_net.start_async(request_id=0, inputs={ self.input_name: frame })
        if self.exec_net.requests[0].wait(-1) == 0:
            pred = self.exec_net.requests[0].outputs[self.output_name]
            return pred

    def check_model(self):
        self.input_name = next(iter(self.model.inputs))
        self.output_name = next(iter(self.model.outputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_shape = self.model.outputs[self.output_name].shape


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        height = self.input_shape[2]
        width = self.input_shape[3]
        frame = cv2.resize(image, (width, height))
        frame = frame.transpose((2, 0, 1))
        return frame

    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        height = image.shape[0]
        width = image.shape[1]
        faces = []
        count = 1
        for box in outputs[0][0]:
            if box[2] >= 0.6:
                x_min = int(box[3] * width)
                y_min = int(box[4] * height)
                x_max = int(box[5] * width)
                y_max = int(box[6] * height)
                cropped_image = image[y_min:y_max, x_min:x_max]
                faces.append(cropped_image)
        return faces

def draw_boxes(pred, image):
    height = image.shape[0]
    width = image.shape[1]
    count = 1
    for box in pred[0][0]:
        if box[2] >= 0.6:
            x_min = int(box[3] * width)
            y_min = int(box[4] * height)
            x_max = int(box[5] * width)
            y_max = int(box[6] * height)
            frame = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 0, 225), thickness=2)
            cropped_image = image[y_min:y_max, x_min:x_max]
            cv2.imwrite(f'cropped_image_{count}.jpg', cropped_image)
            count += 1
    cv2.imwrite('face.jpg', frame)

def main():
    CPU_EXTENSION_MAC = '/opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib'
    model_name = 'models/intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001'
    image = 'bin/test_image2.png'
    model = Model_X(model_name=model_name, device='CPU', extensions=CPU_EXTENSION_MAC)
    model.load_model()
    image = cv2.imread(image)
    pred = model.predict(image)
    faces = model.preprocess_output(pred, image)
    for idx, face in enumerate(faces):
        print(face.shape)
        cv2.imwrite(f'cropped_image{idx}.jpg', face)
    draw_boxes(pred, image)
    print(pred.shape)



if __name__ == '__main__':
    main()