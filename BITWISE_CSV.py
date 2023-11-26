
import cv2
path1 = r'D:\XuLyAnh\IMAGE\blckwht1.png' 
path2 = r'D:\XuLyAnh\IMAGE\blckwht2.png'
img1= cv2.imread(path1)
img2= cv2.imread(path2)
dest_and = cv2.bitwise_and(img2, img1, mask=None)
cv2.imshow('Anh 1', img1) 
cv2.imshow('Anh 2', img1)
cv2.imshow('Anh bitwise and', dest_and)

#0xff = 1111111
if cv2.waitKey(0) & 0xff == 27:  #Mã unicdoe của ESC là 27
    cv2.destroyAllWindows()