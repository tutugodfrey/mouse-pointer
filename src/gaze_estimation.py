'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import sys
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

class GazeEstimation:
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

    def predict(self, inputs):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        inputs = self.preprocess_input(inputs)
        self.exec_net.start_async(request_id=0, inputs=inputs)
        if self.exec_net.requests[0].wait(-1) == 0:
            return self.exec_net.requests[0].outputs[self.output_name]

    def check_model(self):
        self.input_names = list(self.model.inputs.keys())
        self.input_shapes = {}
        for name in self.input_names:
           self.input_shapes[name] = self.model.inputs[name].shape

        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def preprocess_input(self, inputs):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        left_eye_image = inputs['left_eye_image']
        right_eye_image = inputs['right_eye_image']
        head_pose = inputs['head_pose_angles']

        # shape for left and right eyes are the same
        # using the shape of left eye
        height = self.input_shapes['left_eye_image'][2]
        width = self.input_shapes['left_eye_image'][3]
        left_eye_image = cv2.resize(left_eye_image, (width, height))
        left_eye_image = np.expand_dims(left_eye_image.transpose((2, 0, 1)), axis=0)

        right_eye_image = cv2.resize(right_eye_image, (width, height))
        right_eye_image = np.expand_dims(right_eye_image.transpose((2, 0, 1)), axis=0)

        inputs['left_eye_image'] = left_eye_image
        inputs['right_eye_image'] = right_eye_image
        inputs['head_pose_angles'] = head_pose

        return inputs

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        raise NotImplementedError

def main(left_eye, right_eye, head_pose):
    CPU_EXTENSION_MAC = '/opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib'
    gaze_model = 'models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002'

    gaze_estimation = GazeEstimation(model_name=gaze_model, device='CPU', extensions=CPU_EXTENSION_MAC)

    # Load the models
    gaze_estimation.load_model()
    gaze_estimate = gaze_estimation.predict({
        'left_eye_image': left_eye,
        'right_eye_image': right_eye,
        'head_pose_angles': head_pose})

    return gaze_estimate

if __name__ == '__main__':
    main()
