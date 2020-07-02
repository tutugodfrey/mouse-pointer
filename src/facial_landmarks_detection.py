'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import sys
import cv2
from openvino.inference_engine import IENetwork, IECore

class FacialLandmarksDetector:
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
            return self.exec_net.requests[0].outputs[self.output_name], frame

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
        height, width = self.input_shape[2], self.input_shape[3]
        frame = cv2.resize(image, (width, height))

        return frame.transpose((2, 0, 1))

        

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        landmarks_coords = {
            'left_eye': [outputs[0][0], outputs[0][1]],
            'right_eye':[outputs[0][2], outputs[0][3]],
            'nose': [outputs[0][4], outputs[0][5]],
            'left_lip_corner': [outputs[0][6], outputs[0][7]],
            'right_lip_cornet': [outputs[0][8], outputs[0][9]]
        }

        return landmarks_coords

    def get_eyes(self, coords, image):
        image = image.transpose((1, 2, 0))
        height = image.shape[0]
        width = image.shape[1]
        eyes = {}
        for key in coords.keys():
            if key == 'left_eye' or key == 'right_eye':
                x_coord = coords[key][0]
                y_coord = coords[key][1]
                x_min = int(x_coord * (width - 35))
                y_min = int(y_coord * (height - 35))
                x_max = int(x_coord * (width + 35))
                y_max = int(y_coord * (height + 35))
                cropped_image = image[y_min:y_max, x_min:x_max]
                eyes[key] = cropped_image

        return eyes

def main():
    CPU_EXTENSION_MAC = '/opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib'
    model_name = 'models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009'
    image = 'bin/test_image2.png'
    model = FacialLandmarksDetector(model_name=model_name, device='CPU', extensions=CPU_EXTENSION_MAC)
    model.load_model()
    image = cv2.imread(image)
    pred, frame = model.predict(image)
    pred = model.preprocess_output(pred)
    eyes = model.get_eyes(pred, image)
    for eye, cropped_image in eyes.items():
        cv2.imwrite(eye +'_image.jpg', cropped_image)

if __name__ == '__main__':
    main()
