import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO


model = YOLO("yolov8n.pt")

sct = mss()
monitor = sct.monitors[3]  
cv2.namedWindow("YOLOv8 Screen Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Screen Detection", 1200, 800)
drawing = False
start_x, start_y = -1, -1
rectangles = []

def draw_rectangle(event, x, y, flags, param):
    global start_x, start_y, drawing, rectangles

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Draw temporary rectangle on a copy of the frame
            frame_copy = param.copy()
            cv2.rectangle(frame_copy, (start_x, start_y), (x, y), (0, 255, 0), 2)
            # Draw all permanent rectangles
            for rect in rectangles:
                x1, y1, x2, y2 = rect
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imshow("YOLOv8 Screen Detection", frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        rectangles.append((start_x, start_y, end_x, end_y))

# Attach the mouse callback
cv2.setMouseCallback("YOLOv8 Screen Detection", draw_rectangle)

while True:
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # YOLO inference
    results = model(frame, conf=0.3, verbose=False)

    
    annotated_frame = results[0].plot()

    for rect in rectangles:
       x1, y1, x2, y2 = rect
       cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("YOLOv8 Screen Detection", annotated_frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
#