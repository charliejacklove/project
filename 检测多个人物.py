import cv2 as cv


def facedetection():
    gray_image = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    face_detect = cv.CascadeClassifier(
        'C:/Users/93678/PycharmProjects/project/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gray_image, 1.1, 5,0)
    for x, y, w, z in face:
        cv.rectangle(img, (x, y), (x + w, w + z), (0, 0, 255), 2)
    cv.imshow('result', img)


img = cv.imread('youth.jpg')
facedetection()
while True:
    if ord('q') == cv.waitKey(0):
        break;
