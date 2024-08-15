import torch


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