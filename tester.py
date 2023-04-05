import numpy as np
import cv2
import os


cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
        'C:/Users/93678/PycharmProjects/project/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
eyes_cascade =cv2.CascadeClassifier('C:/Users/93678/PycharmProjects/project/venv/Lib/site-packages/cv2/data/haarcascade_mcs_mouth.xml')

while (cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, h, w) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eyes_cascade.detectMultiScale(roi_gray)
        for (ex, ey, eh, ew) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 3)

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if (key == 27):
        break

cap.release()
cv2.destroyAllWindows()
