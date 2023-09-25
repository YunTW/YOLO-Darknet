# YOLOv4 Darknet Real-Time Object Detection

![Python](https://img.shields.io/badge/Python-14354C.svg?logo=python&logoColor=white) ![OpenCV](https://img.shields.io/badge/Opencv-5C3EE8.svg?logo=Opencv&logoColor=white) ![YOLO](https://img.shields.io/badge/YOLO-00FFFF.svg?logo=YOLO&logoColor=black)

![Object Detection](https://miro.medium.com/v2/resize:fit:2792/format:webp/1*Co8xD0IWPaBiWr-Xfu38dw.jpeg)
:link: [YOLOv4 — the most accurate real-time neural network on MS COCO dataset.](https://alexeyab84.medium.com/yolov4-the-most-accurate-real-time-neural-network-on-ms-coco-dataset-73adfd3602fe)

本專案使用兩個方法來實現 YOLO 物件偵測。

1. Darknet 原作者寫的 python API
2. OpenCV DNN Module

## Table of content

- [YOLOv4 Darknet Real-Time Object Detection](#yolov4-darknet-real-time-object-detection)
  - [Table of content](#table-of-content)
  - [Requirement](#requirement)
  - [Installation](#installation)
  - [Folder](#folder)
  - [Usage](#usage)
  - [Contact](#contact)

## Requirement

- Python 3.9.18
- CUDA Toolkit 11.0.3
- cuDNN v8.0.5

## Installation

- Numpy
  
  ```bash
  pip install numpy
  ```

- OpenCV

  ```bash
  pip install opencv-python
  pip install opencv-contrib-python
  ```

## Folder

- cfg: [yolov4.cfg](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg)

- data: [coco.data](https://github.com/AlexeyAB/darknet/blob/master/cfg/coco.data), [coco.names](https://github.com/AlexeyAB/darknet/blob/master/cfg/coco.names)

- Download `yolov4.weights` file: [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)

- yolo: [YOLOv4 Darknet](https://github.com/AlexeyAB/darknet) 編譯過的函式庫

:pushpin: Faster

- cfg: [yolov4-tiny.cfg](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny.cfg)

- Download `yolov4-tiny.weights` file: [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)

## Usage

- Training Example
  [![colab](https://user-images.githubusercontent.com/4096485/86174097-b56b9000-bb29-11ea-9240-c17f6bacfc34.png)](https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg)

- Detection from file
  
  ```bash
  python darknet_file.py
  ```

  ```bash
  python opencv_dnn_file.py
  ```

  ![result](./result.jpg)

- Detection from webcam
  
  ```bash
  python darknet_realtime.py
  ```

  ```bash
  python opencv_dnn_realtime.py
  ```

## Contact

[![Github](https://img.shields.io/badge/Github-100000.svg?logo=github&logoColor=white)](https://github.com/YunTW) [![Linkedin](https://img.shields.io/badge/Linkedin-0077B5.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yuntw/) [![Gmail](https://img.shields.io/badge/Gmail-D14836?logo=gmail&logoColor=white)](terrell60813@gmail.com)
