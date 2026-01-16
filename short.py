#This model does not use yolov8, it uses frame comparison/background modeling (easier on the GPU/better with small camera feeds)
import cv2
import numpy as np
from mss import mss
import json
import os
import pygame

RECT_FILE = "rectangles2.json"
if os.path.exists(RECT_FILE):
    with open(RECT_FILE, "r") as f:
        rectangles = json.load(f)
else:
    rectangles = []


pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")
sct = mss()
monitor = sct.monitors[2]  
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

cv2.setMouseCallback("EyesX", draw_rectangle)
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=25,
    detectShadows=True
)
while True:
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    current_frame = frame.copy()
    
    fgmask = fgbg.apply(frame)

    # Remove shadows (keep only real motion)
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    # Clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    #Draw rectangles
    trigger_alert = False
    MOTION_THRESHOLD = 500
    for rect in rectangles:
       x1, y1, x2, y2 = rect
       cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
       x1, x2 = min(x1, x2), max(x1, x2)
       y1, y2 = min(y1, y2), max(y1, y2)

       roi = fgmask[y1:y2, x1:x2]
       motion_pixels = cv2.countNonZero(roi)

       if motion_pixels > MOTION_THRESHOLD:
            trigger_alert = True
            break

    
    #Red alert
    if trigger_alert:
        red_overlay = np.zeros_like(frame)
        red_overlay[:, :, 2] = 255

        frame = cv2.addWeighted(
            frame, 0.7,
            red_overlay, 0.3,
            0
        )

    cv2.imshow("EyesX", frame)

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

cv2.destroyAllWindows()