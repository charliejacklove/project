import cv2
import numpy as np
import random

face_cascade=  face_detect = cv2.CascadeClassifier(
        'C:/Users/93678/PycharmProjects/project/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
mouth_cascade=cv2.CascadeClassifier('C:/Users/93678/PycharmProjects/project/venv/Lib/site-packages/cv2/data/haarcascade_mcs_mouth.xml')
nose_casade = cv2.CascadeClassifier('C:/Users/93678/PycharmProjects/project/venv/Lib/site-packages/cv2/data/haarcascade_mcs_nose.xml')
font =cv2.FONT_HERSHEY_SIMPLEX
org = (30, 30)

weared_mask_font_color = (0, 255, 0) # 绿色
not_weared_mask_font_color = (0, 0, 255) #  红色
noface = (255, 255, 255) #白色
thickness = 2
font_scale = 1
weared_mask = "Thank You for wearing MASK"
not_weared_mask = "Please wear MASK to defeat Corona"

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
    img = cv2.flip(img, 1)#图像 垂直翻转
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#灰度处理
    (thresh, black_and_white) = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    #继续对图片进行处理抓捕特征值 比如鼻子或者眼睛 处理的图片还是32位 将图像进行分割 提取我们想要的东西
    faces = face_cascade.detectMultiScale(gray, 1.1, 10)
    # 第一个参数是检测的图片 第二个参数是搜素窗口的比列系数 每一次放大百分之10,第三个参数表示每一个目标至少要检测3次才能算是真正的目标，
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 20)
    #深度提取样本的特征值 如面部或者鼻子或者其他部分，至少检测四次找到目标值
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



