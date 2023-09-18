import cv2
from datetime import datetime
import time

from mylib import computer_vision as cv
from mylib.deploy_model import OpenCV_DNN

# define
dnn = OpenCV_DNN("weights/yolov4.weights", "cfg/yolov4.cfg")
# 時間格式
DATE_FORMAT = "%Y_%m_%d_%H_%M_%S"  # 年_月_日_時_分_秒

# webcam
cap, writer, height, width = cv.cam_init(
    0, is_write=False, save_path=f"video{datetime.now().__format__(DATE_FORMAT)}.avi"
)

while cap.isOpened():
    start_time = time.time()
    ret, img = cap.read()  # 取得 WebCam 即時畫面

    if not ret:
        break

    # 物件偵測
    classes, scores, boxes = dnn.detection(img)

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

    # ----FPS calculation
    FPS = cv.calc_fps(start_time)

    # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
    cv2.putText(
        img, f"FPS {FPS}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2
    )

    # 視窗顯示圖像
    cv2.imshow("Object Detection", img)

    # image writing
    if writer is not None:
        writer.write(img)

    k = cv2.waitKey(1)
    if k == ord("q"):  # 關閉
        break
    elif k == ord("s"):  # 儲存影像
        cv2.imwrite(f"image{datetime.now().__format__(DATE_FORMAT)}.jpg", img)

cap.release()
cv2.destroyAllWindows()  # 關閉所有cv2建立的視窗
