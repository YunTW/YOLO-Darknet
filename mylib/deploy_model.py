import cv2
import os
from yolo import darknet


class OpenCV_DNN:
    def __init__(self, weights_path: str, cfg_path: str) -> None:
        """
        Using OpenCV DNN Deploy YOLOv4 Darknet Model

        Parameters:

        weightsPath: 權重檔案路徑/檔名.weights

        cfgPath: 模型參數檔案路徑/檔名.cfg
        """
        self.__weights_path = weights_path
        self.__cfg_path = cfg_path

    @property
    def WeighstPath(self) -> str:
        return self.__weights_path

    @WeighstPath.setter
    def WeighstPath(self, path: str):
        """
        權重檔案路徑
        """
        self.__weights_path = path

    @property
    def CfgPath(self) -> str:
        return self.__cfg_path

    @CfgPath.setter
    def CfgPath(self, path: str):
        """
        模型參數檔案路徑
        """
        self.__cfg_path = path

    def detection(self, img: cv2.Mat, confidence_threshold: float = 0.5) -> tuple:
        """
        物件偵測

        parameters:

        img_path: 圖像路徑/檔名

        return:

        (classes: 類別, scores: 信任分數, boxes: [圖像x, 圖像y, 圖像w, 圖像h])
        """
        NMS_THRESHOLD = 0.4
        # net = cv2.dnn.readNet(self.__weights_path, self.__cfg_path) # OpenCV 深度學習模組
        net = cv2.dnn.readNetFromDarknet(
            cfgFile=self.__cfg_path, darknetModel=self.__weights_path
        )
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)
        classes, scores, boxes = model.detect(img, confidence_threshold, NMS_THRESHOLD)
        return classes, scores, boxes
