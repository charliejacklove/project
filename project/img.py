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

cv2.waitKey(0)


