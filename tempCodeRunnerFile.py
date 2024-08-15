from ultralytics import YOLO
import numpy as np
import cv2

cap = cv2.VideoCapture(1)
model = YOLO('yolov8n.pt')
cap.set(3, 1280)
cap.set(4, 720)

while True:
    ret, frame = cap.read()
    results = model(frame, conf=0.25, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
    cv2.imshow("img", frame)
    cv2.waitKey(1)
