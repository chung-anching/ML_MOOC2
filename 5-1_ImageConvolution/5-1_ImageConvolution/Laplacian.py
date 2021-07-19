import numpy as np

#以下兩種載入方式二擇一，依照OpenCV安裝過程不同，可能需要不同的載入方式
from cv2 import cv2
# import cv2
 
imageName = "lena.jpg"
img = cv2.imread(imageName, cv2.IMREAD_COLOR)

# 設定kernel size為3x3
kernel_size = 3

# Laplacian
kernel1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
kernel2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

print (kernel1)
print (kernel2)

# 使用cv2.filter2D進行convolute，
img1 = cv2.filter2D(img, ddepth=3, dst=-1, kernel=kernel1, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
img2 = cv2.filter2D(img, ddepth=3, dst=-1, kernel=kernel2, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
cv2.imwrite("convoluted1.jpg", img1)
cv2.imwrite("convoluted2.jpg", img2)