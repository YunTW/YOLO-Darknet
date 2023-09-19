# YOLOv4 Darknet Real-Time Object Detection

本專案使用兩個方法來實現 YOLO 物件偵測。

1. Darknet 原作者寫的 python API
2. OpenCV DNN Module

## Requirement

- Python 3.9.18
- CUDA Toolkit 11.0.3
- cuDNN v8.0.5

## Installed

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
- Download `yolov4.weights` file 245 MB: [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
- yolo: [YOLOv4 Darknet](https://github.com/AlexeyAB/darknet) 編譯過的函式庫

## Usage

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
