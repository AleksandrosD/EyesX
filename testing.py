import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import json
import os
# import pygame
muted = False
RECT_FILE = "rectangles.json"
if os.path.exists(RECT_FILE):
    with open(RECT_FILE, "r") as f:
        rectangles = json.load(f)
else:
    rectangles = []


model = YOLO("yolov8n.pt")
# pygame.mixer.init()
# pygame.mixer.music.load("alarm.mp3")
sct = mss()
monitor = sct.monitors[1]  
cv2.namedWindow("EyesX", cv2.WINDOW_NORMAL)
cv2.resizeWindow("EyesX", 1200, 800)
drawing = False
start_x, start_y = -1, -1

current_frame = None

def rectangles_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)

    return not (
        ax2 < bx1 or  
        ax1 > bx2 or  
        ay2 < by1 or  
        ay1 > by2     
    )

def draw_rectangle(event, x, y, flags, param):
    global start_x, start_y, drawing, rectangles, current_frame

    if current_frame is None:
        return
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Draw temporary rectangle on a copy of the frame
            frame_copy = current_frame.copy()
            cv2.rectangle(frame_copy, (start_x, start_y), (x, y), (0, 255, 0), 2)
            # Draw all permanent rectangles
            for rect in rectangles:
                x1, y1, x2, y2 = rect
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imshow("EyesX", frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        rectangles.append((start_x, start_y, end_x, end_y))

# Attach the mouse callback
cv2.setMouseCallback("EyesX", draw_rectangle)

while True:
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    current_frame = frame.copy()

    # YOLO inference
    results = model(frame, conf=0.3,classes=[0, 2, 7], verbose=False)
    annotated_frame = results[0].plot()

    #Draw rectangles
    for rect in rectangles:
       x1, y1, x2, y2 = rect
       cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    #If rectangles overlap 
    trigger_alert = False
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        for rect in rectangles:
            if rectangles_overlap((x1, y1, x2, y2), rect):
                trigger_alert = True
                break

        if trigger_alert:
            break 
    
    #Red alert
    # if muted:
    #     pygame.mixer.music.set_volume(0)
    # else:
    #     pygame.mixer.music.set_volume(1)

    if trigger_alert:
        red_overlay = np.zeros_like(annotated_frame)
        red_overlay[:, :, 2] = 255

        annotated_frame = cv2.addWeighted(
            annotated_frame, 0.7,
            red_overlay, 0.3,
            0
        )
    #     if not pygame.mixer.music.get_busy():
    #         pygame.mixer.music.play()
    # else:
    #     if pygame.mixer.music.get_busy():
    #         pygame.mixer.music.stop()

    words = ["Exit App (q)", "Delete All Boxes (d)", "Mute Alarm (m)"]
    x, y = 10, 30
    line_height = 35  # vertical spacing

    for i, word in enumerate(words):
        cv2.putText(annotated_frame, word,
                    (x, y + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA)

    cv2.imshow("EyesX", annotated_frame)

    # "q" for exit/save, "d" for delete all rectangles
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        # Save rectangles
        with open(RECT_FILE, "w") as f:
            json.dump(rectangles, f)
        break
    elif key == ord("d"):
        rectangles.clear()  # remove from memory

        if os.path.exists(RECT_FILE):
            os.remove(RECT_FILE)  # remove from disk

        print("Rectangles deleted")
    elif key==ord("m"):
        muted = not muted

cv2.destroyAllWindows()