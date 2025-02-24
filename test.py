import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Load class names from coco.txt
with open("coco.txt", "r") as f:
    class_list = f.read().split("\n")

# Define tracking instance
tracker = Tracker()

def mousrePtr(event,x,y,flags,param):
    if event == cv2.EVENT_MOUSEMOVE:
        print([x,y])
    

cv2.namedWindow('Vehicle_count')
cv2.setMouseCallback('Vehicle_count',mousrePtr)
# Open video capture
cap = cv2.VideoCapture('highway.mp4')

# Line crossing detection setup
line_y = 322  # Adjust based on your image
tracked_ids = set()   # Keeps track of seen vehicles
crossed_ids = set()   # Ensures each vehicle is counted only once
incoming_vehicle_count = 0
outgoing_vehicle_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))

    # Perform YOLO detection
    results = model.predict(frame)
    detections = results[0].boxes.data

    # Convert to DataFrame
    px = pd.DataFrame(detections).astype("float")

    vehicle_boxes = []

    # Process detected objects
    for _, row in px.iterrows():
        x1, y1, x2, y2, _, class_id = map(int, row[:6])

        # Consider only vehicle classes (car, truck, bus, motorcycle)
        if class_id in [2, 3, 5, 7]:  # COCO IDs for vehicles
            vehicle_boxes.append([x1, y1, x2, y2])

    # Update tracker
    tracked_objects = tracker.update(vehicle_boxes)

    for obj in tracked_objects:
        x3, y3, x4, y4, obj_id = obj
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2  # Compute centroid

        # Check if the vehicle crossed the line from top to bottom
        if obj_id not in crossed_ids:
            if y3 < line_y and y4 >= line_y:  # Vehicle moved downward across the line
                incoming_vehicle_count += 1
                crossed_ids.add(obj_id)

            elif y3>  line_y and y4 <= line_y:  # Vehicle moved upward across the line
                outgoing_vehicle_count += 1
                crossed_ids.add(obj_id)

        # Draw bounding box and centroid
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(frame, f'ID: {obj_id}', (x3, y3 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Draw the counting line
    cv2.line(frame, (180, line_y), (469, line_y), (0, 0, 255), 2)
    cv2.line(frame, (532, line_y), (795, line_y), (0, 0, 255), 2)

    # Display vehicle counts
    cv2.putText(frame, f'Incoming Vehicles: {incoming_vehicle_count}', (737, 432), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, f'Outgoing Vehicles: {outgoing_vehicle_count}', (738, 479), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("Vehicle_count", frame)

    if cv2.waitKey(0) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
