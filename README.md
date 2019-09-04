# Detector-Assisted-Tracking

This repository introduces Detector-Assisted Tracking: a simple yet efficient and accurate combination between You Only Look Once (YOLO) version 2 (YOLOv2) and OpenCV online trackers (Median Flow (MF), Kernelized Correlation Filter (KCF), Multiple Instance Learning (MIL), Online Boosting (OLB)).

Paper can be found: http://arxiv.org/abs/1908.10406
 -> Submitted to IEEE TBME
 
 In its current form, DAT can track one object at a time. In this case, it can detect either the left or right hand. Changing to different objects is as easy as retraining YOLOv2 (See Step 1 repository). Future work consists of extending to multi-object tracking. 
 
 This code runs in Linux. Uses Python wrapper version of YOLOv2 and OpenCV online trackers in Python
 
# TO-DO
- [x] Single-class implementation
- [ ] Multi-class implementation
- [ ] Make dataset public

# Go-To:
1. [Installation](#installation)
2. [Install OpenCV from source](#how-we-installed-opencv-from-source)
3. [How to Use](#how-to-use)
4. [Notes](#notes)

# Installation

1. Download YOLO repository. I like AlexeyAB/darknet: `git clone https://github.com/AlexeyAB/darknet.git`
2. Download the latest stable version of OpenCV. We used version 3.4.0. [Go to How we installed OpenCV from source](#how-we-installed-opencv-from-source)
3. In cloned YOLO repository, download our code: `git clone https://github.com/RJVisee44/Detector-Assisted-Tracking.git`
4. Download weights file for our model at: 
- Move this into DATModel/

### How we installed OpenCV from source
1. Install the required packages:
- OpenCV: https://opencv.org/releases.html
- OpenCV contrib: https://github.com/opencv/opencv_contrib/releases
     
2. Unpack these and then go into OpenCV directory:
- `cd ~/opencv-3.4.0/`
     
3. `mkdir build && cd build`
  
4. Configure using cmake:
- `cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=/path/to/opencv_contrib-3.4.0/modules/ -D BUILD_EXAMPLES=ON -D WITH_FFMPEG=ON -D WITH_V4L=ON -D WITH_OPENGL=ON -D WITH_CUDA=ON -D CUDA_ARCH_BIN="6.1" -D WITH_CUBLAS=ON -D WITH_CUFFT=ON -D WITH_EIGEN=ON -D BUILD_DOCS=ON ..`
  
5. Build: 
- `make -j8 && sudo make install`

6. Export path
- `sudo gedit ~/.bashrc`
- `export LD_LIBRARY_PATH=/usr/local/lib/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`
- `source ~/.bashrc`

# How to Use
Within DATModel/ is 4 files:
1. model_cfg = "yolov2_DAT.cfg"
2. data_cfg = "yolov2_DAT.data"
3. weights = "yolov2_DAT.weights"
4. names = "yolov2.names"

YOLOv2 weights file was trained on the entire ANS SCI dataset (167,622 frames). Can distinguish between 3 classes. 

If everything is installed, run (example):

`cd Detector-Assisted-Tracking`
`python3 DAT.py --ri 100 --coi 3 --chi 30 --cls 'L' --tracker_type 'KCF' --img_list 'test.txt' --image_path 'ANS SCI/images/'`

Saves output to result_cls.txt

What do all these parameters mean? 

reset_iters (ri):
- The number of frames between each detector usage to reinitialize the tracker and combat against tracker drift. 
- If this parameter was 100, then the detector would be used every 100 frames to reinitialize the tracker or any time the tracker failed.

con_ious (coi):
- The number of consistent detections used to initialize the tracker. 
- If consecutive IOU was 3, then the tracker would be initialized only if the detector found the hand in 3 consecutive frames and every detection had an overlap greater than 0.1 with the previous detection. 
- Also used to disable the tracker if it did not successfully find the hand in the set number of consecutive frames. 

check_iters (chi):
- The number of frames after the tracker was disabled in which the detector attempted to locate the hand. 
- If check iterations was 60, then every 60 frames after the tracker was disabled the detector checked to see if the hand existed. 
- If in that 60th frame the detector was able to locate the hand then the detector attempted to reinitialize the tracker. The tracker remained disabled if the detector was unable to locate the hand. 

cls:
- The object in which you want to detect. In this case either left hand: "L", right hand: "R", or other hands: "O".

tracker_type:
- Online tracker used for tracking. Either "KCF", "MF", "MIL", or "OLB". KCF/MF are the recommended trackers. 

img_list:
- .txt file containing image filenames without an extensions or paths. 

image_path:
- path to test images in img_list. 


# Notes
darknet.so is created when "make" is performed on [Go to Installation](#installation) Step 1. The file you generate may need to be copied into Detector-Assisted-Tracking/ folder
