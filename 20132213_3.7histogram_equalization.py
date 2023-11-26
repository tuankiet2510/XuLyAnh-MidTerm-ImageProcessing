import matplotlib.pyplot as plt
import numpy as np
import cv2

# Function to convert color image to luminance
def rgb_to_luminance(image):
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

# Function to compute histogram equalization
def histogram_equalization_corrected(luminance): #Thực hiện quá trình cân bằng histogram cho hình ảnh mức độ sáng.
    # Convert luminance to integer type
    luminance_int = luminance.astype('uint8') #chuyển đổi giá trị mức độ sáng sang kiểu số nguyên để có thể tính histogram.
    
    # Compute histogram and CDF for the integer luminance values
    hist, bins = np.histogram(luminance_int.flatten(), 256, [0, 256]) #Tính histogram và hàm phân phối tích lũy (CDF) từ những giá trị mức độ sáng này.
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    # Compute the transfer function
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # Apply transfer function to the luminance values
    equalized_luminance = cdf[luminance_int]

    # Normalize the equalized luminance for further processing
    equalized_luminance_normalized = equalized_luminance / 255

    return equalized_luminance_normalized, cdf_normalized

# Load the uploaded image
#image = cv2.imread('D:\\XuLyAnh\\IMAGE\\history_src_img.jpg')
#image = cv2.imread('D:\\XuLyAnh\\IMAGE\\noised_2.jpg')
#image = cv2.imread('D:\\XuLyAnh\\IMAGE\\noised_image.jpg')
#image = cv2.imread('D:\\XuLyAnh\\IMAGE\\low_quality_img.png')
#image = cv2.imread('D:\\XuLyAnh\\IMAGE\\aa.jpg')
image = cv2.imread('D:\\XuLyAnh\\IMAGE\\moon.png')
#image = cv2.imread('D:\\XuLyAnh\\IMAGE\\cthead-16bitABC\\PNG\\cthead-16bit071.png')

#image = cv2.imread('D:\\XuLyAnh\\IMAGE\\ca-chua-1.jpg')
#image =  cv2.imread('D:\\XuLyAnh\\IMAGE\\kkk.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to luminance
luminance = rgb_to_luminance(image_rgb)

# Apply corrected histogram equalization
equalized_luminance_normalized, cdf_normalized = histogram_equalization_corrected(luminance)

# Re-map the color pixels based on the equalized luminance
# Multiplying the normalized RGB values by the equalized luminance
equalized_image = image_rgb * equalized_luminance_normalized[:, :, np.newaxis]
equalized_image = np.clip(equalized_image, 0, 255).astype('uint8')

# Display the original and equalized images
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(equalized_image)
plt.title('Histogram Equalized Image')

plt.show()

# Display the histogram and the cumulative distribution function
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.hist(luminance.flatten(), bins=256, color='gray', alpha=0.75)
plt.title('Original Luminance Histogram')

plt.subplot(1, 2, 2)
plt.plot(cdf_normalized, color='black')
plt.title('Cumulative Distribution Function')

plt.show()
