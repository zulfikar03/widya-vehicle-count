import cv2
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