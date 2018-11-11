import cv2
import sys

# python resize_image.py 縦のピクセル　横のピクセル
image_tate=int(sys.argv[1]) 
image_yoko=int(sys.argv[2])

img = cv2.imread('image/menkyo2.jpg')
img = cv2.resize(img , (image_yoko, image_tate)   )
cv2.imwrite('new_image/menkyo.jpg',img)