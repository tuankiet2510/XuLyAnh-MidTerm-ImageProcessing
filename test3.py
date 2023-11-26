import cv2
import matplotlib.pyplot as plt
def adjust_bilateral_filter(image, diameter, sigmaColor, sigmaSpace):
    if diameter == 1:
        return image
    else:
        adjusted = cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)
        return adjusted
#img1 = cv2.imread('D:\\XuLyAnh\\IMAGE\\aa.jpg')
img1 = cv2.imread('D:\\XuLyAnh\\IMAGE\\noised_image.jpg')
img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = adjust_bilateral_filter(img,9,75,75)
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
    #Open cv đọc ảnh theo định dạng BGR
    #Matplotliv hiển thị theo định dạng RGB
plt.imshow(img)
plt.title('Original Image')
    
plt.subplot(1, 2, 2)
plt.imshow(img2)
plt.title(' Separable filter')
plt.show()