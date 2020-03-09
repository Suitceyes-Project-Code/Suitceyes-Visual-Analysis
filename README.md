# Description

This is the repository for the Suitceyes-Visual Analysis module. In visual analysis or VA, the incoming messages that contain live images from a remote camera are processed and analyzed. The analysis consists of two main areas, which are object detection and classification and face detection and classification. The results from the aforementioned operations can be available locally and distributed through a messaging bus that handles the communication between the camera, the VA service and potentially any module that would require those results. These results consist of JSON files that describe the detected objects or human faces, their position and the class that they belong. In order to utilize the aforementioned platform capabilities of the module a subscribtion to the said message bus (VA_KBS_channel) is required potentially enabling up to thousands simultaneous subscriptions.

## Requirements - Dependencies

The Visual analysis module is developed in Python version 3.5. Below the additional dependencies are listed. Also in the document requirements.txt the additional libraries that will have to be intalled are also listed.

[Realtime framework](https://framework.realtime.co/messaging/ ): A free and scalable cloud-hosted Pub/Sub real-time messaging broker for web and mobile apps. 

[Tensorflow object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection): An open source framework built on top of TensorFlow that helps the construction, training and deployment of object detection models.

[Facenet library](https://github.com/davidsandberg/facenet): A TensorFlow implementation of the face recognizer.

[CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit ): a development environment for creating high performance GPU-accelerated applications.


## Instructions

1. Download and install the Tensorflow Object Detection Api. You can see [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md ) detailed instructions.
2. Clone the project to your local directory.
3. [Download](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.gz ) and place in the models/oid_objects directory, the frozen graph for object detection. 
4. [Download](https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view) and place in the models/20180402-114759 directory, the pretrained model for face detection.
5. Download and install Lampp,
6. Place the upload-service folders in the htdocs directory which is located in your Lampp directory.
7. Update storage address and links in accordance with your IP and desired save location, the to-be-updated links are in the listener, analyzer and index scripts.
8. Download and install CUDA in addition to the Python libraries listed in the requirements.txt document.
9 Start lampp service and run python3 listener.py.

### Platform Capabilities

To utilize the module as a platform and obtain the VA results that include the JSON format output, subscribe to the VA_KBS_channel provided by the Realtime Framework. For more information on the Realtime Framework, you can see [here](https://framework.realtime.co/messaging/#documentation). The output of the VA analysis contains basic information about the image (dimensions, timestamp and name), details about the objects that it found, which are the type of object with the corresponding confidence, and its relative position in the image and in the same fashion human faces and their positions which are also recognised with the assistance of the Facenet library. To access the original images, as they are being transmitted, a subsription to the VA_RS_channel is required. 
## Contact 

For further details, please contact Elias Kouslis (kouslis@iti.gr).

