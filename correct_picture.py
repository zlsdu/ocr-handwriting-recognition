import cv2
import numpy as np

#读取图像，做二值化处理
img = cv2.imread('img/tilt.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
#像素取反，变成白字黑底
# gray = cv2.bitwise_not(gray)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow('thresh', thresh)

#计算包含了旋转文本的最小边框
coords = np.column_stack(np.where(thresh > 0))
print(coords)
#该函数给出包含着整个文字区域矩形边框，这个边框的旋转角度和图中文本的旋转角度一致
angle = cv2.minAreaRect(coords)[-1]
print(angle)

#调整角度
if angle < -45:
    angle = -(90+ angle)
else:
    angle = -angle

#仿射变换
h, w = img.shape[:2]
center = (w//2, h//2)
print(angle)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
cv2.putText(rotated, 'Angle: {:.2f} degrees'.format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

print('[INFO] angel :{:.3f}'.format(angle))
cv2.imshow('Input', img)
cv2.imshow('Rotated', rotated)