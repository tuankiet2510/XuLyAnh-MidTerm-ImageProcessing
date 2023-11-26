import cv2
path = r'D:\XuLyAnh\IMAGE\RGB_IMG.png'
image = cv2.imread(path)
B, G, R = cv2.split(image)
cv2.imshow('Anh goc', image)
cv2.imshow('Blue',B)
cv2.imshow('Red',R)
cv2.imshow('Green',G)
cv2.waitKey()