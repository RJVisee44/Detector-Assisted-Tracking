# Detector-Assisted-Tracking

This repository introduces Detector-Assisted Tracking: a simple yet efficient and accurate combination between You Only Look Once (YOLO) version 2 (YOLOv2) and OpenCV online trackers (Median Flow (MF), Kernelized Correlation Filter (KCF), Multiple Instance Learning (MIL), Online Boosting (OLB)).

Paper can be found: http://arxiv.org/abs/1908.10406
 -> Submitted to IEEE TBME
 
 In its current form, DAT can track one object at a time. In this case, it can detect either the left or right hand. Changing to different objects is as easy as retraining YOLOv2 (See Step 1 repository). Future work consists of extending to multi-object tracking. 
 
 This code runs in Linux. Uses Python wrapper version of YOLOv2 and OpenCV online trackers in Python
 
# Installation

1. Download YOLO repository. I like AlexeyAB/darknet: `git clone https://github.com/AlexeyAB/darknet.git`
2. Download the latest stable version of OpenCV. We used version 3.4.0. [Go to How we installed OpenCV from source](#how-we-installed-opencv-from-source)
3. In cloned YOLO repository, download our code: `git clone https://github.com/RJVisee44/Detector-Assisted-Tracking.git`

### How we installed OpenCV from source
1. Install the required packages:
- OpenCV: https://opencv.org/releases.html
- OpenCV contrib: https://github.com/opencv/opencv_contrib/releases
     
2. Unpack these and then go into OpenCV directory:
- `cd ~/opencv-3.4.0/`
     
3. `mkdir build && cd build`
  
4. Configure using cmake:
- `cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=/home/anslab/opencv_contrib-3.4.0/modules/ -D BUILD_EXAMPLES=ON -D WITH_FFMPEG=ON -D WITH_V4L=ON -D WITH_OPENGL=ON -D WITH_CUDA=ON -D CUDA_ARCH_BIN="6.1" -D WITH_CUBLAS=ON -D WITH_CUFFT=ON -D WITH_EIGEN=ON -D BUILD_DOCS=ON ..`
  
5. Build: 
- `make -j8 && sudo make install`

6. Export path
- `sudo gedit ~/.bashrc`
- `export LD_LIBRARY_PATH=/usr/local/lib/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`
- `source ~/.bashrc`


