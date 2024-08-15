import cv2
import numpy as np
import torch
from src.detector import YoloDetector
from src.counting import track_and_count_vehicles
from deep_sort_realtime.deepsort_tracker import DeepSort

# buka file video
video_path = "https://github.com/zulfikar03/widya-vehicle-count/raw/main/assets/toll_gate.mp4"  # Change this to your video path
cap = cv2.VideoCapture(video_path)

# Dapatkan dimensi video
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Definisikan VideoWriter untuk menyimpan video dalam format .avi
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output_video\\result_video.avi', fourcc, 30, (width, height))

# menginisiasi YOLOv5 detector
detector = YoloDetector()

# Garis lurus gerbang tol
limits = [17, 151, 568, 284]

# Variabel untuk menyimpan total kendaraan yang terdeteksi
totalCount_car = []
totalCount_bus = []
totalCount_truck = []

# mendefinisikan gerbang dari kiri ke kanan sebanyak 8 
gates = [[14, 157, 74, 174],
    [74, 174, 138, 190],
    [138, 190, 203, 203],
    [203, 203, 271, 218],
    [271, 218, 339, 232],
    [411, 248, 485, 264],
    [485, 264, 561, 284]]

# inisiasi object tracker
car_tracker = DeepSort()
bus_tracker = DeepSort()
truck_tracker = DeepSort()

# Looping menjalankan program
while cap.isOpened():
    # Membuka video
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Membuat frame aman jika terjadi error
    if frame is None:
        continue
    
    #Dapatkan hasil prediksi dari YoloDetector untuk mendapatkan detections[(x1, y1, w, h), confidence, class_label]
    results = detector.score_frame(frame)
    _,detections_car = detector.plot_boxes('car', results, frame, height=frame.shape[0], width=frame.shape[1], confidence=0.5)
    _,detections_bus = detector.plot_boxes('bus', results, frame, height=frame.shape[0], width=frame.shape[1], confidence=0.5)
    _,detections_truck = detector.plot_boxes('truck', results, frame, height=frame.shape[0], width=frame.shape[1], confidence=0.5)
    print(detections_truck)
    
    # Hitung dan bbox mobil
    bboxes_car, totalCount_car = track_and_count_vehicles(frame, detections_car, car_tracker, totalCount_car, 'car', gates, limits)
    print(bboxes_car)
    # Hitung dan bbox bus
    bboxes_bus, totalCount_bus = track_and_count_vehicles(frame, detections_bus, bus_tracker, totalCount_bus, 'bus', gates, limits)
    print(bboxes_bus)
    # Hitung dan bbox truk
    bboxes_truck, totalCount_truck = track_and_count_vehicles(frame, detections_truck, truck_tracker, totalCount_truck, 'truck', gates, limits)
    print(bboxes_truck)
    # Visualisasi bounding box masing-masing object
    for (x1, y1, x2, y2, track_id) in bboxes_car:
        cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 5, (255, 0, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id} car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    for (x1, y1, x2, y2, track_id) in bboxes_bus:
        cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 5, (255, 0, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'ID: {track_id} bus', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    for (x1, y1, x2, y2, track_id) in bboxes_truck:
        cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 5, (255, 0, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f'ID: {track_id} truck', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    
    #Menampilkan informasi kendaraan yang terhitung
    totalVehicles = len(totalCount_car)+len(totalCount_bus)+len(totalCount_truck)
    # Tampilkan jumlah kendaraan yang telah dihitung
    cv2.putText(frame, f'Total Vehicles: {totalVehicles}', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(frame, f'Car: {len(totalCount_car)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(frame, f'Bus: {len(totalCount_bus)}', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(frame, f'Truck: {len(totalCount_truck)}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Simpan frame ke video output
    out.write(frame)
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()