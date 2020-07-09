#! /bin/bash

echo 'Starting Mouse Pointer Controller!
Be sure to set an interval of about 1 sec between moves to achieve best result'

source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
INPUT_TYPE=$1
echo $INPUT_TYPE

if [[ $INPUT_TYPE == '' ]]; then # use default setting
  python3 src/inference_pipeline.py

elif [[ $INPUT_TYPE == 'cam' ]]; then
  python3 src/inference_pipeline.py -i cam -lm FP32 -gz FP32 -hd FP32 -p low -s medium -vh 1 -vg 1 -vl 1

elif [[ $INPUT_TYPE == 'video' ]]; then
  python3 src/inference_pipeline.py -i ./bin/demo.mp4 -lm FP32 -gz FP32 -hd FP32 -p low -s medium -vh 1 -vg 1 -vl 1

elif [[ $INPUT_TYPE == 'image' ]]; then
  python3 src/inference_pipeline.py -i ./bin/test_image1.jpg -lm FP32 -gz FP32 -hd FP32 -p low -s medium -vh 1 -vg 1 -vl 1
fi
