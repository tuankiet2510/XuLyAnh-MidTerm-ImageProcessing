import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

# Function to convert color image to luminance
def rgb_to_luminance(image):
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

# Function to distribute histogram values to vertices
def distribute_to_vertices(hist, width, height):
    #vertex_hist = np.zeros((height + 1, width + 1))
    vertex_hist = np.zeros((height, width, 256))
    for i in range(height):
        for j in range(width):
            # Distribute to four vertices
            vertex_hist[i, j] += hist[i, j] / 4
            vertex_hist[i, (j + 1) % width] += hist[i, j] / 4
            vertex_hist[(i + 1) % height, j] += hist[i, j] / 4
            vertex_hist[(i + 1) % height, (j + 1) % width] += hist[i, j] / 4
            '''
            vertex_hist[i, j + 1] += hist[i, j] / 4
            vertex_hist[i + 1, j] += hist[i, j] / 4
            vertex_hist[i + 1, j + 1] += hist[i, j] / 4
            '''
    return vertex_hist

# Function to compute the local histogram for each patch
def compute_local_histogram(image, patch_size):
    '''
    #demension = image.shape
    height = image.shape[0]
    width = image.shape[1]


    histograms = np.zeros((height // patch_size, width // patch_size, 256))
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            hist, _ = np.histogram(patch, bins=256, range=(0, 256))
            histograms[i // patch_size, j // patch_size, :] = hist
    '''
    height, width = image.shape[:2]  # Lấy chiều cao và chiều rộng của hình ảnh

    # Tính toán số lượng patch theo chiều cao và chiều rộng
    num_patches_height = height // patch_size
    num_patches_width = width // patch_size

    # Khởi tạo mảng histograms để lưu trữ các histogram của patch
    histograms = np.zeros((num_patches_height, num_patches_width, 256))

    # Lặp qua từng patch
    for i in range(num_patches_height):
        for j in range(num_patches_width):
            # Tính toán vị trí của patch trong hình ảnh gốc
            patch_x_start = j * patch_size
            patch_x_end = (j + 1) * patch_size
            patch_y_start = i * patch_size
            patch_y_end = (i + 1) * patch_size

            # Trích xuất patch từ hình ảnh gốc
            patch = image[patch_y_start:patch_y_end, patch_x_start:patch_x_end]

            # Tính histogram của patch và lưu vào mảng histograms
            hist, _ = np.histogram(patch, bins=256, range=(0, 256))
            histograms[i, j, :] = hist
    return histograms

# Function to convert histograms to CDFs and perform optional low-pass filtering
def histograms_to_cdfs(histograms, use_filtering=True):
    cdfs = np.cumsum(histograms, axis=2)
    cdfs = cdfs / cdfs[:, :, -1][:, :, np.newaxis]  # Normalize CDFs
    if use_filtering:
        # Apply a Gaussian filter for smoothing
        cdfs = gaussian_filter(cdfs, sigma=1.0)
    return cdfs

# Function to interpolate CDFs for final lookup
def interpolate_cdfs(cdfs, luminance, patch_size):
    interpolated = np.zeros_like(luminance)
    height, width = luminance.shape
    for i in range(height):
        for j in range(width):
            x, y = j // patch_size, i // patch_size
            x = x % cdfs.shape[1]  # Áp dụng phép chia lấy dư cho x
            y = y % cdfs.shape[0]  # Áp dụng phép chia lấy dư cho y
            x_weight, y_weight = j % patch_size / patch_size, i % patch_size / patch_size
            # Bilinear interpolation
            top_left = cdfs[y, x]
            '''
            top_right = cdfs[y, x + 1]
            bottom_left = cdfs[y + 1, x]
            bottom_right = cdfs[y + 1, x + 1]
            top = top_left * (1 - x_weight) + top_right * x_weight
            bottom = bottom_left * (1 - x_weight) + bottom_right * x_weight
            interpolated_value = top * (1 - y_weight) + bottom * y_weight
            # Map the luminance value using the interpolated CDF
            pixel_value = luminance[i, j]
            interpolated[i, j] = interpolated_value[pixel_value] '''
            top_right = cdfs[y, (x + 1) % (width // patch_size)]
            bottom_left = cdfs[(y + 1) % (height // patch_size), x]
            bottom_right = cdfs[(y + 1) % (height // patch_size), (x + 1) % (width // patch_size)]
            top = top_left * (1 - x_weight) + top_right * x_weight
            bottom = bottom_left * (1 - x_weight) + bottom_right * x_weight
            interpolated_value = top * (1 - y_weight) + bottom * y_weight
            # Map the luminance value using the interpolated CDF
            pixel_value = luminance[i, j]
            interpolated[i, j] = np.interp(pixel_value, np.arange(256), interpolated_value)
    return interpolated

#LOAD IMG
#image_path = 'D:\\XuLyAnh\\IMAGE\\moon.png'
image_path = 'D:\\XuLyAnh\\IMAGE\\cthead-16bitABC\\PNG\\cthead-16bit071.png'
#image_path = 'D:\\XuLyAnh\\IMAGE\\noised_image.jpg'
image = cv2.imread(image_path)
demension = image.shape
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
luminance = rgb_to_luminance(image_rgb)

# Define the patch size
patch_size = 16  # Choose a suitable patch size

# Compute local histograms and distribute counts to vertices
local_histograms = compute_local_histogram(image, patch_size)
#print(local_histograms)
vertex_histograms = distribute_to_vertices(local_histograms, demension[1] // patch_size, demension[0]// patch_size)

# Compute CDFs for each vertex and perform optional low-pass filtering
vertex_cdfs = histograms_to_cdfs(vertex_histograms)

# Interpolate CDFs for final lookup
equalized_luminance = interpolate_cdfs(vertex_cdfs, luminance, patch_size)

# Rescale the luminance values and apply them to the RGB image
equalized_luminance_rescaled = (equalized_luminance / equalized_luminance.max() * 255).astype('uint8')
final_equalized_image = image_rgb.astype('float32')
for i in range(3):
     # Kiểm tra xem giá trị là không hợp lệ
    #invalid_values = np.isnan(equalized_luminance_rescaled) | np.isinf(equalized_luminance_rescaled)
    
    # Xử lý giá trị không hợp lệ bằng cách gán giá trị mặc định (ví dụ: 0)
    #equalized_luminance_rescaled[invalid_values] = 0
    
    # Thực hiện phép nhân
    final_equalized_image[:, :, i] *= equalized_luminance_rescaled / (luminance + 1e-6)

final_equalized_image = np.clip(final_equalized_image, 0, 255).astype('uint8')

# Display the original and equalized images
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(final_equalized_image)
plt.title('Locally Histogram Equalized Image')
plt.show()
