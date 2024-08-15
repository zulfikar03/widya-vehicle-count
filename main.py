import cv2
import numpy as np
import torch

from deep_sort_realtime.deepsort_tracker import DeepSort

# Creating a class for object detection which plots boxes and scores frames in addition to detecting an 
# object

class YoloDetector():

    def __init__(self):
        #Using yolov5s for our purposes of object detection, you may use a larger model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using Device: ', self.device)
    
    def score_frame(self, frame):
        self.model.to(self.device)
        
        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        return labels, cord
    
    def class_to_label(self, x):
        return self.classes[int(x)]
    
    def plot_boxes(self, label_name, results, frame, height, width, confidence=0.3):

        labels, cord = results
        detections = []

        n = len(labels)
        x_shape, y_shape = width, height

        for i in range(n):
            row = cord[i]

            if row[4]>=confidence:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                
                #In this demonstration, we will only be detecting persons. You can add classes of your choice
                if self.class_to_label(labels[i]) == label_name:

                    confidence = float(row[4].item())

                    detections.append(([x1, y1, int(x2-x1), int(y2-y1)], confidence, label_name))
        
        return frame, detections

# Fungsi untuk melacak dan menghitung kendaraan berdasarkan jenisnya
def track_and_count_vehicles(frame, detections, tracker, totalCount, class_name, gates, limits):
    # Gunakan DeepSORT untuk melacak objek berdasarkan deteksi YOLOv8
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # List untuk menyimpan bounding box
    bboxes = []
    
    # Loop untuk menggambar bounding boxes dan ID objek pada frame
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()  # bounding box format (left, top, right, bottom)
        x1, y1, x2, y2 = map(int, ltrb)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        # Simpan bounding box dan track_id untuk visualisasi di luar
        bboxes.append((x1, y1, x2, y2, track_id))
        
        # Update deteksi kendaraan yang melewati gerbang
        for i, gate in enumerate(gates, start=1):
            # Cek apakah kendaraan berada dalam batas gerbang
            if gate[0] < cx < gate[2] and gate[1] - 15 < cy < gate[1] + 15:
                # Jika kendaraan belum ada dalam daftar, tambahkan
                if totalCount.count(track_id) == 0:
                    totalCount.append(track_id)
                    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (255, 0, 255), 5)

    # Kembalikan bounding box dan totalCount
    return bboxes, totalCount

# Open the video file
video_path = "toll_gate (2).mp4"  # Change this to your video path
cap = cv2.VideoCapture(video_path)

# Dapatkan dimensi video
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Definisikan VideoWriter untuk menyimpan video dalam format .avi
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('result_video.avi', fourcc, 30, (width, height))

#Initializing the detection class
detector = YoloDetector()

# Definisikan batas gerbang
limits = [17, 151, 568, 284]

# Variabel untuk menyimpan total kendaraan yang terdeteksi
totalCount_car = []
totalCount_bus = []
totalCount_truck = []

# Definisikan posisi gerbang
gates = [[14, 157, 74, 174],
    [74, 174, 138, 190],
    [138, 190, 203, 203],
    [203, 203, 271, 218],
    [271, 218, 339, 232],
    [411, 248, 485, 264],
    [485, 264, 561, 284]]

#Initialise the object tracker class
car_tracker = DeepSort()
bus_tracker = DeepSort()
truck_tracker = DeepSort()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Ensure frame is valid before proceeding
    if frame is None:
        continue

    results = detector.score_frame(frame)
    _,detections_car = detector.plot_boxes('car', results, frame, height=frame.shape[0], width=frame.shape[1], confidence=0.5)
    _,detections_bus = detector.plot_boxes('bus', results, frame, height=frame.shape[0], width=frame.shape[1], confidence=0.5)
    _,detections_truck = detector.plot_boxes('truck', results, frame, height=frame.shape[0], width=frame.shape[1], confidence=0.5)
    print(detections_truck)
    #tracks_car = object_tracker.update_tracks(detections_car, frame=frame)
    #tracks_bus = object_tracker.update_tracks(detections_bus, frame=frame) 
    #track_truck = object_tracker.update_tracks(detections_truck, frame=frame)

          # Hitung mobil
    bboxes_car, totalCount_car = track_and_count_vehicles(frame, detections_car, car_tracker, totalCount_car, 'car', gates, limits)
    print(bboxes_car)
    # Hitung bus
    bboxes_bus, totalCount_bus = track_and_count_vehicles(frame, detections_bus, bus_tracker, totalCount_bus, 'bus', gates, limits)
    print(bboxes_bus)
    # Hitung truk
    bboxes_truck, totalCount_truck = track_and_count_vehicles(frame, detections_truck, truck_tracker, totalCount_truck, 'truck', gates, limits)
    print(bboxes_truck)
    # Visualisasi bounding box dan informasi di luar fungsi
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
    
    totalVehicles = len(totalCount_car)+len(totalCount_bus)+len(totalCount_truck)
    # Tampilkan jumlah kendaraan yang telah dihitung
    cv2.putText(frame, f'Total Vehicles: {totalVehicles}', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(frame, f'Car: {len(totalCount_car)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(frame, f'Bus: {len(totalCount_bus)}', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(frame, f'Truck: {len(totalCount_truck)}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Simpan frame ke video output
    out.write(frame)
    cv2.imshow('Frame', frame)
    # NOTE: Bounding box expects to be a list of detections, each in tuples of ([left, top, w, h], confidence, detection class)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()