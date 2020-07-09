# Computer Pointer Controller

The Computer Pointer Controller application currently use Four (4) AI models to estimate the direction of gaze of the eyes and uses it to move the mouse of your computer. The Models uses for the application are highlighted below

1. **Face Detector Model**: Will detect face/faces in an image, video, and camera stream. The face detected is the cropped out and passed to intermediate model describe below

2. **Facial Landmark Detection Model**: This MOdel uses the face detected from the face detector model to find/ detect facial landmarks in the face (right eye, left eye, nose, right lip corner, left lip corner). Our interest here is the right and left eyes. These are cropped out and passed to the gaze estimation model describe below.

3. **Head Pose Estimation**: The Head pose estimation model use the face detected from the face detection model to estimate the position or direction of the head. From this model we get the yaw, pitch, roll of the head which will be feed to the gaze estimation model

4. **Gaze Estimation Model**: The gaze estimation model use the prediction from the facial landmark detection model (left eye, right eye) along with the prediction from the Head pose estimation model (yaw, pitch and roll) to estimate the direction of gaze of the eyes.

The prediction from gaze estimation model is then use as input to pyautogui package to control the mouse of your computer.

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

### Enter the project directory
`cd mouse_pointer`

### Setup a virtual environment
`python3 -m venv env`

### Activate the environment
`source env/bin/activate`

### Install required packages
`pip install -r requirement.txt`

### Source the openvino environment
`source /opt/intel/openvino/bin/setupvars.sh`

### Make a directory to save the models
`mkdir models`

### Downloading the required models
- ```python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name landmarks-regression-retail-0009 -o models --cache_dir .```

- ```python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name head-pose-estimation-adas-0001 -o models --cache_dir .```

- ```python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name face-detection-adas-binary-0001 --precision FP32-INT1 -o models --cache_dir .```

- ```python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name gaze-estimation-adas-0002 -o models --cache_dir .```


## Demo
There are two possible ways to run the application currently. 
1. Run the inference_pipeline.py file directly from the commandline passing in disired arguments to modify the behavior and performance of the application. If you simple want to use the default setting the command below will run the application. But remember you can pass argument to the command to change the application behavior.

`python3 inference_pipeline.sh`

2. Use the run_app.sh file provided at the root of the project to run the application. The run_app.sh file accepts 0 or 1 argument that tells the type of input file to feed to the model value are `cam`, `video`, `image` or no argument at all to use the defualt settings in the application. Example is shown below.

`./run_app.sh cam`

To see all possible arguments that can modify the behavior of the application run Python3 inference_pipeline.sh --help.

It is easy to simply modify the run_app.sh file to your desired setting for running the application

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.
#### File structure

mouse_pointer/ : project root
            ./bin/

                ./demo.mp4 - A demo video 

            ./src/  - project source files directory

                  ./facial_landmarks_detection.py contains the code for handling the facial landmark detection model

                  ./head_pose_estimation.py - contains the code to handling the head pose estimation model

                  ./input_feeder.py - contains the code to load inputs that will be passed to the models

                  ./mouse_controller.py - contains the code to move the mouse using the prediciton from the gaze estimation model

                  ./utils.py - helper file for post post processing and handling output from the model (like visualising the output)

                  ./face_detection.py  - contains the code for handling the face detection model

                  ./gaze_estimation.py - contains code for handling the gaze estimation model

                  ./inference_pipeline.py

                  ./run_inference.py - file that uses the models to perform the actual inference on input frames

            ./models/intel - models directory

            ./run_app.sh - bash script for running inference

            .requirements.txt - contains python packages required for running the application

            ./README.md  - project description

            ./env/ - if you have created a virtual environment following the instructions above, you should have this directory





To get the best of the application, there are various parameters that could be modified to change the behavior and performance of the application. All parameter that could be set have safe defaults such that you do not need to pass in any argument to get the application to work. If you need to change the behavior of the application, you can see the arguments to modify by running the help command below

`python3 inference_pipeline.py --help`

Below is a description of various argument that passed to the model and how to visualize the output of the application


**Input** Acceptable input types are images, video, and 'cam' for camera. use the `-i /path/to/file` to pass input image or video. use `-i cam` to use camera stream. The default value is `./bin/demo.mp4` avaliable in the project root directory.

**Setting Model Precision:** This application assumes that the models .xml and .bin files are in the models directory, with their difference precisions.
`./models/intel/model_name/precision/model_name.{xml,bin}`. When this is the case. You can change the precision for each model by passing commandline argument as follows.

`-lm` to pass the desired precision for the facial landmark detection. e.g `-lm FP32`

`-gz` to pass the desired precision for the gaze estimation model. e.g `-gz FP32`

`-hd` to pass the sried precision for the head pose estimation model e.g `-gz FP32`

`-fd` to pass the desired precision for the face detection model e.g `-fd INT1` INT one is the only precision available at time of developing this application, so it the default value.

**Device** This is the device to use for running inference like `CPU`, `VPU`. To pass a desired device use the `-d` flag e.g `-d CPU`. `CPU` is the default value.

**Threshold** The threshold use the confidence level to use with the face detection model for getting faces in a frame. To pass threshold to the application use the `-t` flag e.g `-t 0.6`. `0.6` is the default value.

**CPU extension** The CPU extension to use for handling unsupported layers in the inference engine. Use the `-x` flag to pass the path to the CPU extension. The path to the CPU extension for MacOS is `/opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib` which is set as the default value.

**Visualizing the output of intermediate models:** There are several ways to visualize the output of intermediate models depending on what your focus is. Each of the models have flag set to allow you visualize the outputs or your might just what to see everyone of them. The argument for visualizing the output are described below. The default value for each is `0` if you simple what to use the application without caring about the output. 

`-v 1` To see all possible output from the models.

`-vf 1` Visualize output from the face detection model. this will crop out the detected face from the original image, it will also draw a bounding pose of the detected face in the original image

`-vh 1` to visualize the output from the head pose estimation model

`-vl 1` to visualize the output from the landmark estimation model. You'll see the landmarks drawn on the frames.

`-vg 1` to visualize the gaze estimation model. Draws an arraw line in the eyes pointing in the direction of gaze.

`-vo 1` this will visualize a combination of output from intermediate model in one original image

**Controlling the mouse pointer:** There are two flag to control the mouse pointer

`-p` for precision. This will determine how far the mouse should move acceptable values are `high`, `low`, `medium`. The default value is `low`. Low (low) precision mean the mouse will move very far and high is the otherwise.

`-s` for speed. This will determine how fast the mouse should move. Accepteable values are `fast`, `slow`, `medium`. The default value is `fast`





## Benchmarks
The tables below show for load time and inference time for each model with difference precisions.
The result is achieve by using the video at bin/demo.mp4 width a 0.5 second delay  between execution of frame.
Total frames processed is `595`

**Model sizes**

| Name                  | Size of FP32 | Size of PF16 | Size of INT  |
|-----------------------|--------------|--------------|--------------|
| Face Detection        |       x      |  x           | 2.3 MBb      |
| Facial Landmark Model | 807 KB       |  414 KB      |       x      |
| Head Pose Estimation  | 7.7 MB       |  4.2 MB      |       x      |
| Gaze Estimation       | 7.7 MB       |  4 MB        | 7.8 MB       |
|                       | -            |  -           |       -      |


**Model Load time and Inference time at glance**

| Name                  | Load T. FP32 | Load T. PF16 | Load T. INT  | AVG Infer T. FP32 | AVG Infer T  FP16 |  AVG Infer T. INT  |
|-----------------------|--------------|--------------|--------------|-------------------|-------------------|------------------- |
| Face Detection        |       x      |  x           | 0.21873      |      x            | x                 | 0.01578            |
| Facial Landmark Model | 0.05322      |  0.06014     |       x      | 0.00055           | 0.00056           | x                  |
| Head Pose Estimation  | 0.07082      |  0.12871     |       x      | 0.00178           | 0.00184           | x                  |
| Gaze Estimation       | 0.08480      |  0.10630     | 0.12865      | 0.00210           | 0.00218           | 0.00170            |
|                       | -            |  -           |       -      |                   |                   |                    |


#### Facial landmank compare FP32 and FP16 over 10 interaction (inference time)

**Diff = FP32 - FP16**

| FP16              | FP32              | Diff              |
|-------------------|-------------------|-------------------|
| 0.00057           | 0.00057           | 0.00000           |
| 0.00056           | 0.00058           | 0.00002           |
| 0.00057           | 0.00057           | 0.00000           |
| 0.00055           | 0.00055           | 0.00000           |
| 0.00056           | 0.00054           | -0.00002          |
| 0.00057           | 0.00055           | -0.00002          |
| 0.00055           | 0.00217           | 0.00162           |
| 0.00056           | 0.00062           | 0.00006           |
| 0.00057           | 0.00055           | -0.00002          |
| 0.00059           | 0.00058           | -0.00001          |
|                   |                   |                   |

#### Head pose estimation compare FP32 and FP16 over 10 iteraction (inference time)

**Diff = FP32 - FP16**

| FP16              | FP32              | Diff              |
|-------------------|-------------------|-------------------|
| 0.00182           | 0.00185           | 0.00003           |
| 0.00183           | 0.00180           | -0.00003          |
| 0.00188           | 0.00183           | -0.00005          |
| 0.00178           | 0.00185           | 0.00007           |
| 0.00185           | 0.00180           | -0.00005          |
| 0.00182           | 0.00188           | 0.00006           |
| 0.00181           | 0.00187           | 0.00006           |
| 0.00185           | 0.00194           | 0.00009           |
| 0.00183           | 0.00184           | 0.00001           |
| 0.00188           | 0.00182           | -0.00006          |
|                   |                   |                   |


#### Gaze estimation compare FP32 and FP16 over 10 iteration (inference time)

**Diff = FP32 - FP16**

| FP16              | FP32              | Diff              |
|-------------------|-------------------|-------------------|
| 0.00215           | 0.00222           | 0.00007           |
| 0.00219           | 0.00211           | -0.00008          |
| 0.00222           | 0.00211           | -0.00011          |
| 0.00208           | 0.00214           | 0.00006           |
| 0.00214           | 0.00211           | -0.00003          |
| 0.00211           | 0.00217           | 0.00006           |
| 0.00210           | 0.00217           | 0.00007           |
| 0.00216           | 0.00222           | 0.00006           |
| 0.00218           | 0.00222           | 0.00004           |
| 0.00223           | 0.00215           | -0.00008          |
|                   |                   |                   |


#### Facial landmark compare FP32 and FP16 over 10 interaction (load time)

**Diff = FP32 - FP16**

| FP16              | FP32              | Diff              |
|-------------------|-------------------|-------------------|
| 0.04986           | 0.03975           | -0.01011          |
| 0.03989           | 0.04448           | 0.00459           |
| 0.03954           | 0.04616           | 0.00662           |
| 0.03905           | 0.03803           | -0.00102          |
| 0.04861           | 0.03984           | -0.00877          |
| 0.05489           | 0.04096           | -0.01393          |
| 0.05629           | 0.06214           | 0.00585           |
| 0.03973           | 0.08487           | 0.04514           |
| 0.05757           | 0.06782           | 0.01025           |
| 0.03792           | 0.05296           | 0.01504           |
|                   |                   |                   |


#### Head pose estimation compare FP32 and FP16 over 10 iteraction (load time)

**Diff = FP32 - FP16**

| FP16              | FP32              | Diff              |
|-------------------|-------------------|-------------------|
| 0.09077           | 0.06270           | -0.02807          |
| 0.07323           | 0.07208           | -0.00115          |
| 0.09617           | 0.07044           | -0.02573          |
| 0.07011           | 0.06093           | -0.00918          |
| 0.09558           | 0.05943           | -0.03615          |
| 0.07708           | 0.08883           | 0.01175           |
| 0.07209           | 0.06222           | -0.00987          |
| 0.07289           | 0.11115           | 0.03826           |
| 0.09723           | 0.08927           | -0.00796          |
| 0.07352           | 0.07078           | -0.00274          |
|                   |                   |                   |


#### Gaze estimation compare FP32 and FP16 over 10 iteration (load time)

**Diff = FP32 - FP16**

| FP16              | FP32              | Diff              |
|-------------------|-------------------|-------------------|
| 0.08417           | 0.09539           | 0.01122           |
| 0.07773           | 0.07878           | 0.00105           |
| 0.11229           | 0.08990           | -0.02239          |
| 0.08060           | 0.07071           | -0.00989          |
| 0.10142           | 0.07939           | -0.02203          |
| 0.08910           | 0.08635           | -0.00275          |
| 0.08396           | 0.06995           | -0.01401          |
| 0.07785           | 0.08487           | 0.00702           |
| 0.13349           | 0.10750           | -0.02599          |
| 0.08384           | 0.08888           | 0.00504           |
|                   |                   |                   |

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

The benchmark results above shows model sizes, load time and inference time. I will embark on discussion some of this reasons for the differences in this section.

**Model Size:**
As we can see from the chart above, the sizes of the model differs depending on whether it FP32 and FP16, INT8. FP32 models are generally much larger compare to FP16 because of the extra space required to store bits. We can also see that int8 is much larger than FP32 for similar reason.

**Load Time:**
To benchmark the load time of the models I track the load time for then consecutive executions as shown in the result above for both FP32 and FP16. Though the time was not done alternatively, we can see that there some difference in the load time for FP32 and FP16 with FP32 taking more time to load. That is to be expected loading that the model sizes and noting that FP32 model are generally larger that FP16.

**Inference Time:**
The recording above is the average time for infering 595 total frame from the demo video with 0.5 second interval between processing frames. I like to point out again that frame processing is handled differently for video and camera as the use of video is simple be for experimentation and testing purpose. That said, we see that are some marginal/ negligible difference in the inference time for FP32 and FP16. The general trend is that FP32 takes less time to infer compare to FP16. Since the comparison was not done alternatively, it hard to tell how the two compare. But if we take a look at the single run in the first table we can see that FP32 takes less time to infer compare to FP16 we can observe this behavior. This was not what I was expectng at first, so I research on it a bit. One possible reason that FP32 takes less time to run would be because intel CPU are built for 32/64 bits. While CPU now support 16 bits they internally scale to 32 bits (See reference below) The scaling time might very much contribute to the overall inference time for FP16. While the difference in inference time might be negligigle, it can add up  and impact the overvall performance of our systems hence the need to always consider where there are bottlenecks in the application to do proper optimization.

**Under Resulting IR precision**
[Optimization Guide](https://docs.openvinotoolkit.org/latest/_docs_optimization_guide_dldt_optimization_guide.html)

**Comparing load time and Inference time of the different models:** 

Observation from the tables above also help us to see that the size of the model, the number of operations have an effect in the load time and inference time. For example, the face detection model takes significantly more time to infer compared to the other models because the image size it accepts is much larger that the others. We can also observe that gaze estimation model and head pose estimation model takes more time to load compare to facial landmark model because they are larger in size.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.
N/A

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.
N/A

### Edge Cases

In this section I detail edge cases and consideration that are made to ensure better performance of the applications.

1. **Determing When to Move Mouse Pointer**: While building the application, I could observe that irrespective of whether you move your eyes and head, the model will make certain prediction. If we decide to move the mouse base on every prediction we will only observe random moves on the screen. To this end I find a way to cut out some of the model prediction so that they do not have effect on the mouse move.
  *  One way I achieve is to handle the inference job for camera and video in a slightly different way. For video also frame as processed, but only frames process with a 0.5 seconds interval is passed to the mouse controller. For camera stream, not all frames are processed I set a 1 second time interval for the model to make prediction on input stream. All frame between the 1 second interval is completely ignored. This means that user should attempt to move their eyes or head at I second interval. All of these help to avoid random move of the mouse pointer.

**NOTE:** The consideration to handle the video and camera stream differently is because video will be use mostly for testing the application. If we let all the frame pass without processing the frame will finish very quickly and we will not have much move of the mouse. Whereas when using camera, the camera can be on for as long as the user wants. 

  * Another refining step is to estimate the difference between the previous prediction and the current prediction. If the difference is not that much, I perhap means that it is a random prediction by the models in that case the mouse is not moved. I also consider a situation where the is a significant move in the difference in the y-direction or x-direction but low  difference in the other. In such a case the mouse is set to move in the direction where there is much difference. This allows us to achieve real vertical and horizontal moves of the mouse.

2. **Muse pointer locking at the edges of the screen:** When the prediction by the model moves the mouse to any particular direction, the mouse pointer position is reset by pyautogui. When there is a consistent prediction in a particular direction and the mouse moves to that direction and reaches the edge of the screen, the mouse pointer position keeps increasing above the screen size, but the mouse can not move further in that direction so it locks at the edge. Any further prediction in the opposite direction will be trying to offset the position of the mouse pointer. In most cases, the value is too high that it could never be offset, and the mouse pointer will just stay locked at the edge.
  **Remedy** To prevent this from happen, I set condition to detect when the mouse pointer position is greater than the screen size the particular direction and reset it position to be at the very edge (x-min=0, y-min=0, x-max=screen_size_x, y-max=screen_size_y). The way the mouse pointer positions will alway remain at the edge it value exceed the edge of the screen

3. **More than one person in the frame:** It is possible for more that one person to be in the screen, or image. If that is the case the application will use only the first face detected.

**Lighting:** For effective use of the application on camera stream, it is important that there be adequate lighting in operational environment. If there is not enough lighting the application will simple exit reporting that 'No more frame to read' and the camera stream will stop.
