import cv2
path_apple = r'D:\XuLyAnh\IMAGE\transparent-apple-21.png'
#path_apple = r'D:\XuLyAnh\IMAGE\apple_trans.jpg'
path_green = r'D:\\XuLyAnh\\IMAGE\\green_plain_bg.jpg'
apple1 = cv2.imread(path_apple)
#apple = cv2.resize(apple1, (720,720))
green = cv2.imread(path_green)

green[80:980, 80:980] = apple1
cv2.imshow('green',green)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()