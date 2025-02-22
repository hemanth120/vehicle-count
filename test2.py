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

# Define polygon areas for vehicle counting
Line_x1,Line_y1=444,388
Line_x2,Line_y2=158,320
# COCO class IDs for different vehicles
vehicle_classes = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Initialize counters
incoming_counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}
outgoing_counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}
tracked_ids = {}

# Mouse callback function to get pixel coordinates
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print([x, y])

cv2.namedWindow('CV')
cv2.setMouseCallback('CV', RGB)

# Open video capture
cap = cv2.VideoCapture('highway.mp4')

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:  # Process every 3rd frame for efficiency
        continue

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

        if class_id in vehicle_classes:
            vehicle_boxes.append([x1, y1, x2, y2, class_id])

    # Update tracker
    tracked_objects = tracker.update(vehicle_boxes)

    for obj in tracked_objects:
        x3, y3, x4, y4, obj_id, class_id = obj
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2  # Compute centroid
        vehicle_type = vehicle_classes.get(class_id, "Unknown")

        if obj_id not in tracked_ids:
            tracked_ids[obj_id] = vehicle_type

        # Check if the vehicle is in the incoming lane
        if cv2.pointPolygonTest(np.array(incoming_lane, np.int32), (cx, cy), False) >= 0:
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f'ID: {obj_id} {vehicle_type}', (x3, y3 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            if obj_id not in tracked_ids:
                incoming_counts[vehicle_type] += 1

        # Check if the vehicle is in the outgoing lane
        if cv2.pointPolygonTest(np.array(outgoing_lane, np.int32), (cx, cy), False) >= 0:
            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
            cv2.putText(frame, f'ID: {obj_id} {vehicle_type}', (x3, y3 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            if obj_id not in tracked_ids:
                outgoing_counts[vehicle_type] += 1

    # Draw polygon areas
    cv2.polylines(frame, [np.array(incoming_lane, np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(frame, [np.array(outgoing_lane, np.int32)], True, (0, 0, 255), 2)

    # Display vehicle counts
    y_offset = 432
    for vehicle, count in incoming_counts.items():
        cv2.putText(frame, f'Incoming {vehicle}: {count}', (737, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_offset += 30

    y_offset = 479
    for vehicle, count in outgoing_counts.items():
        cv2.putText(frame, f'Outgoing {vehicle}: {count}', (738, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_offset += 30

    cv2.imshow("CV", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
