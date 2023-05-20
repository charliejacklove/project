import cv2
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
img=cv2.imread('blackimg.jpg')
img = cv2.flip(img, 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
result=cv2.medianBlur(img,7)

(ret1,imgs)=cv2.threshold(gray,80,255,cv2.THRESH_BINARY)
(ret2,img2)=cv2.threshold(gray,90,255,cv2.THRESH_BINARY)
(ret3,img3)=cv2.threshold(gray,100,255,cv2.THRESH_OTSU)
a=cv2.equalizeHist(gray)
plt.figure()
plt.subplot(221),plt.hist(a.ravel(),256)

plt.subplot(222),plt.hist(gray.ravel(),256)

plt.show()

cv2.imshow('source img',a)
cv2.imshow("mediamBlur",img)
face_cascade=  face_detect = cv2.CascadeClassifier(
        'C:/Users/93678/PycharmProjects/project/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
mouth_cascade=cv2.CascadeClassifier('C:/Users/93678/PycharmProjects/project/venv/Lib/site-packages/cv2/data/haarcascade_mcs_mouth.xml')
nose_casade = cv2.CascadeClassifier('C:/Users/93678/PycharmProjects/project/venv/Lib/site-packages/cv2/data/haarcascade_mcs_nose.xml')
font =cv2.FONT_HERSHEY_SIMPLEX
org = (30, 30)

weared_mask_font_color = (0, 255, 0) # GREEN
not_weared_mask_font_color = (0, 0, 255) # RED
noface = (255, 255, 255) #WHITE
thickness = 2
font_scale = 1
weared_mask = "Thank You for wearing MASK"
not_weared_mask = "Please wear MASK "

cap = cv2.VideoCapture(0)

def nose_detection(img):
    img=cv2.flip(img,1)
    gray_image=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    (thr,black)=cv2.threshold(gray_image,880,255,cv2.THRESH_MASK)
    Nose=nose_casade.detectMultiScale(gray_image,1.1,3)
    Noses=nose_casade.detectMultiScale(black,1.1,5)
    if (len(Nose) == 0 and len(Noses) == 0):
        cv2.putText(img,"",org,font,font_scale,noface,thickness,cv2.LINE_AA)
    elif(len(Nose)==0 and len(Noses)==1):
        cv2.putText(img,weared_mask,org,font,font_scale,weared_mask_font_color,thickness,cv2.LINE_AA)
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)


def mouse_detection(img):
    img=cv2.flip(img,1)
    gray_images=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    (t,ha)=cv2.threshold(gray_images,880,255,cv2.THRESH_MASK)
    mouth=mouth_cascade.detectMultiScale(gray_images,1.1,3)
    mouths=mouth_cascade.detectMultiScale(ha,1.1,6)
    if (len(mouths)==0 and len(mouths)==0):
        cv2.putText(img,"",org,font,font_scale,noface,thickness,cv2.LINE_AA)
    elif(len(mouths)==0 and len(mouths)==1):
        cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(img, not_weared_mask, org, font, font_scale, not_weared_mask_font_color, thickness,
                        cv2.LINE_AA)
            break


while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)#Image Vertical flip
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#Grayscale processing
    (thresh, black_and_white) = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    #Continuing to process the image to capture feature values such as a nose or eyes, the image is still processed in 32 bits, and the image is segmented to extract what we want.
    faces = face_cascade.detectMultiScale(gray, 1.1, 10)
    # The first parameter is the image to be detected The second parameter is the scale factor of the search window 10 percent for each zoom, and the third parameter indicates that each target must be detected at least 3 times before it can be considered a real target.
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 20)
    #Deep extraction of feature values of the sample e.g. face or nose or other parts, at least four times to find the target value
    flag_nose=nose_detection(img)
    mouse_detection(img)

    if (len(faces) == 0 and len(faces_bw) == 0):
        cv2.putText(img, "No face found", org, font, font_scale, noface, thickness, cv2.LINE_AA)
    elif (len(faces) == 0 and len(faces_bw) == 1):
        cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)
        if (len(mouth_rects) == 0):
            cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
        else:
            for (mx, my, mw, mh) in mouth_rects:
                if (y < my < y + h):
                    cv2.putText(img, not_weared_mask, org, font, font_scale, not_weared_mask_font_color, thickness,
                                cv2.LINE_AA)
                    break
    cv2.imshow('Mask Detection', img)
    key=cv2.waitKey(1000//100)
    if key==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()



