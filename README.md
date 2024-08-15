# PROJECT TOLL GATES VEHICLE TRACKING AND COUNTING

## PROJECT OVERVIEW

This project is to detect cars, buses, and trucks that are traveling through the toll gate. When passing through the toll gate, the system will count the number of cars, buses, and trucks so that the total number of vehicles is calculated. By using the YOLOv5 algorithm to detect vehicles and Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric) for tracking. There are 8 gates that are drawn straight lines so that when a vehicle passes through it will be seen so that it is counted.

## PROJECT STRUCTURE
- `.idea/`: Configuration directory for the IDE.
- `assets/`: Folder for storing assets needed in the project.
- `notebook/`: Folder for storing notebooks containing explanations of using YOLOv8.
- `output_video/`: This folder contains the output video of the vehicle counting process.
- `src/`: The folder containing the main source code of the project.
- `YOLO-weights/`: The folder that stores the YOLO model
- `.gitignore`: Configuration file for Git, which defines files and folders that are not tracked by Git.
- `classes.txt`: A text file that lists the class labels used by the YOLO model.
- `main.py`: The main script to run the vehicle counting program.
- `README.md`: This document.
- `requirements.txt`: A file containing the Python dependencies required by the project.
- `yolov5s.pt`: The trained YOLOv5 model.
- `yolov8n.pt`: The trained YOLOv8 model.

## INSTALLATION
1. **Requirements**
 ``sh
   python 3.10
   ```
2. **Creating a Virtual Environment**
```sh
    conda create -p venv python==3.10
```
3. **Clone a Github Repository**
 ```sh
   git clone https://github.com/zulfikar03/widya-vehicle-count.git
   cd vehicle-count-widya
   ```
4. Install Dependencies Used**
``` sh
    pip install -r requirements.txt
```
5. **Run the Main Script**
```sh
    python main.py
```



