# PROJECT TOLL GATES VEHICLE TRACKING AND COUNTING

## PROJECT OVERVIEW

Projek ini adalah untuk mendeteksi objek kendaraan mobil, bus, dan truk yang sedang berjalan melewati gerbang tol. Saat melewati gerbang tol maka sistem akan menghitung jumlah mobil, bus, dan truk sehingga jumlah total kendaraan terhitung. Dengan menggunakan algoritma YOLOv5 untuk mendeteksi kendaraannya dan Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric) untuk tracking. Terdapat 8 gerbang yang ditarik garis lurus sehingga ketika terlewati kendaraan akan terlihat sehingga terhitung.

## PROJECT STRUCTURE
- `.idea/`: Direktori konfigurasi untuk IDE.
- `assets/`: Folder untuk menyimpan aset yang diperlukan dalam proyek.
- `output_video/`: Folder ini berisi video output hasil proses penghitungan kendaraan.
- `src/`: Folder yang berisi kode sumber utama proyek.
- `YOLO-weights/`: Folder yang menyimpan model YOLO
- `.gitignore`: File konfigurasi untuk Git, yang mendefinisikan file dan folder yang tidak di-track oleh Git.
- `classes.txt`: File teks yang berisi daftar class label yang digunakan oleh model YOLO.
- `main.py`: Skrip utama untuk menjalankan program penghitungan kendaraan.
- `README.md`: Dokumen ini.
- `requirements.txt`: File yang berisi dependensi Python yang diperlukan oleh proyek.
- `yolov5s.pt`: Model YOLOv5 yang telah dilatih.
- `yolov8n.pt`: Model YOLOv8 yang telah dilatih.


