import os
from yolo import darknet
import time
import cv2

# # 測試資料集路徑
# test_path = 'folder path'

# test_images = [f for f in os.listdir(test_path) if f.endswith('.jpg')]

# # 隨機圖像路徑
# import random
# img_path = test_path + random.choice(test_images);
img_path = "dog.jpg"

# Parameters
cfg_file = "cfg/yolov4.cfg"  # 模型配置
data_file = "data/coco.data"  # 資料集路徑
weight_file = "weights/yolov4.weights"  # 權重
thre = 0.5
show_coordinates = True

# Load Network
network, class_names, class_colors = darknet.load_network(
    cfg_file, data_file, weight_file, batch_size=1
)

# 取得神經網路的輸入寬高 (Get Nets Input dimentions)
width = darknet.network_width(network)
height = darknet.network_height(network)

# Fix image format
frame = cv2.imread(img_path)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_resized = cv2.resize(frame_rgb, (width, height))

# convert to darknet format, save to “ darknet_image “
darknet_image = darknet.make_image(width, height, 3)
darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

# record start time
start_time = time.time()

# inference
detections = darknet.detect_image(network, class_names, darknet_image, thresh=thre)
darknet.print_detections(detections, show_coordinates)  # 將資訊顯示在終端機上
darknet.free_image(darknet_image)  # 將圖片給清除

# record end time
end_time = time.time()
cost_time = end_time - start_time
print(f"耗時: {cost_time}s")

# draw bounding box
image = darknet.draw_boxes(detections, frame_resized, class_colors)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 儲存圖像
cv2.imwrite("result.jpg", image)


cv2.imshow("result", image)
cv2.waitKey(0)
