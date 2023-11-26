import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_filters(image_path):
    # Load the image
    image1 = cv2.imread(image_path)
    if image1 is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    
    # Convert image to grayscale
    image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur
    gaussian_blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply Median Filter
    median_blur = cv2.medianBlur(gray_image, 5)
    
    # Apply Bilateral Filter
    bilateral_filter = cv2.bilateralFilter(gray_image, 9, 75, 75)
    
    # Apply Sharpening
    kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(gray_image, -1, kernel_sharpening)
    
    # Display the images
    titles = ['Original Image', 'Gaussian Blurring', 'Median Blurring', 'Bilateral Filter', 'Sharpened Image']
    images = [image, gaussian_blur, median_blur, bilateral_filter, sharpened_image]
    
    plt.figure(figsize=(15, 10))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])  # Hide tick marks
    plt.show()

# Replace 'image_path.jpg' with your image file
#apply_filters('D:\\XuLyAnh\\IMAGE\\noised_image.jpg')
apply_filters('D:\\XuLyAnh\\IMAGE\\noised_2.jpg')