from ultralytics import YOLO
import cv2
import numpy as np
import pygame
import time

# ===== INIT =====
model = YOLO("fire.pt")
cap = cv2.VideoCapture(0)

pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")

prev_fire = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fire_detected = False
    frame = cv2.resize(frame, (640, 480))

    # ===== YOLO =====
    results = model(frame, conf=0.3) 

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]

            if label == "fire":
                fire_detected = True
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)

    # ===== COLOR =====
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 120, 120])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 200: 
            fire_detected = True
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)

    # ===== ÂM THANH + TERMINAL =====
    if fire_detected and not prev_fire:
        print("🔥 FIRE DETECTED !!!")
        pygame.mixer.music.play(-1)  # lặp vô hạn

    if not fire_detected and prev_fire:
        print("✅ FIRE CLEARED")
        pygame.mixer.music.stop()  # tắt âm thanh

    prev_fire = fire_detected

    # ===== HIỂN THỊ =====
    if fire_detected:
        cv2.rectangle(frame, (0,0), (640,60), (0,0,0), -1)
        cv2.putText(frame, "🔥 FIRE DETECTED !!!",
                    (50,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 3)

    cv2.imshow("Fire Detection System", frame)



    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()