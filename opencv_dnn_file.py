import cv2
from datetime import datetime
import time

from mylib import computer_vision as cv
from mylib.deploy_model import OpenCV_DNN

# define
dnn = OpenCV_DNN("weights/yolov4.weights", "cfg/yolov4.cfg")
# 時間格式
DATE_FORMAT = "%Y_%m_%d_%H_%M_%S"  # 年_月_日_時_分_秒

# 從資料夾隨機讀取圖像
# img = cv.select_random_image_from_folder('C:/Users/user/Downloads/Baby-Detection.v1i.darknet/test/')
img = cv2.imread("dog.jpg")

# 計算運行時間-起始
start_time = time.time()

# 物件偵測
classes, scores, boxes = dnn.detection(img)

# 計算運行時間-結束
end_time = time.time()
elapsed_time = end_time - start_time  # 單位:秒
print(f"物件偵測耗時: {elapsed_time} s")

# 物件列表
objs = []
classes_list = []

# 類別名稱
with open("data/coco.names", mode="r", encoding="utf-8") as file:
    for line in file:  # 一行一行讀取
        classes_list.append(line)

for i in range(len(classes)):
    objs.append(
        {"class": classes_list[classes[i]], "score": scores[i], "box": boxes[i]}
    )

# 多物件繪製
for obj in objs:
    img = cv.draw_obj_rec(img, obj["class"], obj["score"], obj["box"])

cv2.imshow("Object Detection", img)  # 視窗顯示圖像

k = cv2.waitKey(0)  # 程式停下等待輸入
if k == ord("s"):  # 儲存影像
    cv2.imwrite(f"image{datetime.now().__format__(DATE_FORMAT)}.jpg", img)

cv2.destroyAllWindows()  # 關閉所有cv2建立的視窗
