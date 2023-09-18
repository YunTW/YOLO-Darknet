import os
from yolo import darknet
import time
import cv2

# var
frame_count = 0
fps = 0

# Parameters
cfg_file = "cfg/yolov4.cfg"  # 模型配置
data_file = "data/coco.data"  # 資料集路徑
weight_file = "weights/yolov4.weights"  # 權重
thre = 0.25
show_coordinates = True

# Load Network
network, class_names, class_colors = darknet.load_network(
    cfg_file, data_file, weight_file, batch_size=1
)

# Get Nets Input dimentions
width = darknet.network_width(network)
height = darknet.network_height(network)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()  # 取得 WebCam 即時畫面

    if not ret:
        break

    # record start time
    start_time = time.time()

    # Fix image format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))

    # convert to darknet format, save to “ darknet_image “
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

    # inference
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thre)
    darknet.print_detections(detections, show_coordinates)  # 將資訊顯示在終端機上
    darknet.free_image(darknet_image)  # 將圖片給清除

    # draw bounding box
    image = darknet.draw_boxes(detections, frame_resized, class_colors)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # record end time
    end_time = time.time()
    cost_time = end_time - start_time
    print(f"耗時: {cost_time}s")

    # ----FPS calculation
    fps = int(1 / (time.time() - start_time))

    # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
    cv2.putText(
        image, f"FPS {fps}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2
    )

    cv2.imshow("Object Detection", image)

    k = cv2.waitKey(1)

    if k == ord("q"):  # 關閉
        break

cv2.destroyAllWindows()  # 關閉所有cv2建立的視窗
cap.release()
