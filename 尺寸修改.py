import cv2 as cv
img=cv.imread('wegame.jpg')
#修改尺寸
resize_img=cv.resize(img,(200,200))

#显示原图
cv.imshow('img',img)
#显示修改的图片
cv.imshow('resize_image',resize_img)

print('未修改的照片',img.shape)
print('修改后的照片',resize_img.shape)
cv.waitKey(0)

