# PROJECT TOLL GATES VEHICLE TRACKING AND COUNTING

## PROJECT OVERVIEW

This project aims to detect and count vehicles such as cars, buses, and trucks traveling through This project aims to detect and count vehicles such as cars, buses, and trucks traveling through toll gates. The system records the number of vehicles by type as they pass through the gates. The technologies used in this project include:

- **YOLOv5**: An object detection algorithm to identify vehicles.
- **Deep SORT (Simple Online and Realtime Tracking)**: Used for accurate vehicle tracking to ensure each vehicle is counted only once.

The project incorporates 8 virtual toll gates represented by straight lines. When a vehicle crosses one of these lines, the system detects and counts it based on its category (car, bus, truck), producing a total count for each type.

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
```sh
   python 3.10
```
2. **Creating a Virtual Environment**
```sh
    conda create -p venv python==3.10
    conda activate venv
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



