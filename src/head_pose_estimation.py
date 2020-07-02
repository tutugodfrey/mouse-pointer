'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import sys
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

class HeadPoseEstimation:
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
            self.output_coords = {output_name: self.exec_net.requests[0].outputs[output_name].astype(float).item()
                                for output_name in self.output_names}
            return self.output_coords, frame

    def check_model(self):
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_names = list(self.model.outputs.keys())
        self.output_dict = {}
        for name in self.output_names:
            self.output_dict[name] = self.model.outputs[name].shape  
      

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
        pred = np.array([outputs])
        return pred


def main():
    CPU_EXTENSION_MAC = '/opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib'
    model_name = 'models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001'
    image = 'bin/test-image1.jpg'
    model = HeadPoseEstimation(model_name=model_name, device='CPU', extensions=CPU_EXTENSION_MAC)
    model.load_model()
    image = cv2.imread(image)
    pred = model.predict(image)
    pred = model.preprocess_output(pred)

if __name__ == '__main__':
    main()
