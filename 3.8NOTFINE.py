import numpy as np
import cv2
from scipy.interpolate import interp2d

#Tính toán độ sáng (luminance) của hình ảnh
def compute_luminance(image):
    #
    #using the formula Y = 0.299 * R + 0.587 * G + 0.114 * B.
    res = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    return res
#Tính kích thước batch (patch size) tối ưu cho việc xử lý hình ảnh theo từng phần (patch).
def calculate_batch_size(image_shape, max_batch_size=64, min_patch_size=8):
    """
    Calculate an optimal batch size for processing an image in patches, based on its dimensions.
    
    Args:
    - image_shape (tuple): A tuple representing the shape of the image (height, width).
    - max_batch_size (int): Maximum allowable batch size.
    - min_patch_size (int): Minimum size for each patch.

    Returns:
    - int: An optimal batch size.
    """
    height, width = image_shape

    # Start with the largest possible patch size that is less than or equal to both dimensions of the image
    for patch_size in range(min(max(height, width), max_batch_size), min_patch_size - 1, -1):
        if height % patch_size == 0 and width % patch_size == 0:
            return patch_size

    # If no patch size is found that divides both dimensions, return the smallest patch size
    return min_patch_size

# Example usage
image_shape = (1080, 1920)  # Example image dimensions
optimal_batch_size = calculate_batch_size(image_shape)
optimal_batch_size

#Tính kích thước batch (patch size) tối ưu cho việc xử lý hình ảnh theo từng phần (patch).
def compute_histograms(image, patch_size):
    """
    Divide the image into patches and compute histograms for each patch.
    """
    rows, cols = image.shape
    histograms = np.zeros((rows, cols, 256))
    # lặp qua hình ảnh theo từng phần nhỏ dựa trên patch_size và tính histogram cho mỗi phần.
    for i in range(0, rows, patch_size):
        for j in range(0, cols, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            hist, _ = np.histogram(patch, bins=256, range=(0, 255))
            histograms[i:i+patch_size, j:j+patch_size, :] = hist
    
    return histograms

#Phân phối giá trị histogram đến các đỉnh kề nhau bằng nội suy song tuyến.
def distribute_values_to_vertices(histograms, patch_size):
    """
    Distribute histogram values to adjacent vertices using bilinear interpolation.
    """
    rows, cols, _ = histograms.shape
    vertices = np.zeros((rows+1, cols+1, 256))
    # mỗi giá trị trong histogram, hàm này phân bổ giá trị đó đến các đỉnh kề theo tỷ lệ dựa trên khoảng cách.
    for i in range(rows):
        for j in range(cols):
            for k in range(256):
                value = histograms[i, j, k]
                vertices[i:i+2, j:j+2, k] += value * (1 / patch_size**2)
    
    return vertices


def convert_to_cdf(vertices): #Chuyển đổi histogram tại mỗi đỉnh thành hàm phân phối tích lũy (cumulative distribution function - CDF).
    """
    Convert histograms at each vertex to cumulative distribution functions (CDFs).
    """
    return np.cumsum(vertices, axis=2) #tính tổng lũy kế của giá trị histogram tại mỗi đỉnh để tạo ra CDF.


def low_pass_filter_cdfs(cdfs): #Áp dụng bộ lọc thấp lên CDFs để làm mịn chúng.
    """
    Apply a low-pass filter to the CDFs at each vertex.
    """
    filtered_cdfs = np.zeros_like(cdfs)
    for i in range(cdfs.shape[2]):
        #CDFs được làm mịn bằng cách sử dụng bộ lọc Gaussian, giúp giảm nhiễu và làm mượt dữ liệu.
        filtered_cdfs[:, :, i] = cv2.GaussianBlur(cdfs[:, :, i], (5, 5), 0)
    
    return filtered_cdfs

def interpolate_cdfs(cdfs, image): # Nội suy CDFs kề cận để tạo ra hàm tra cứu cuối cùng.
    """
    Interpolate adjacent CDFs for final lookup.
    """
    rows, cols = image.shape
    equalized_image = np.zeros_like(image)
    
    for i in range(rows):
        for j in range(cols):
            x = [i, i+1]
            y = [j, j+1]
            #pixel_value = image[i, j]
            pixel_value = int(min(max(image[i, j], 0), 255))
            interpolated_cdf = interp2d(x, y, cdfs[i:i+2, j:j+2, pixel_value], kind='linear')
            equalized_value = interpolated_cdf(i, j)
            equalized_image[i, j] = equalized_value
    
    return equalized_image

def local_histogram_equalization(image, patch_size=8):
    """
    Perform local histogram equalization on an image.
    """
    luminance = compute_luminance(image)
    histograms = compute_histograms(luminance, patch_size)
    vertices = distribute_values_to_vertices(histograms, patch_size)
    cdfs = convert_to_cdf(vertices)
    filtered_cdfs = low_pass_filter_cdfs(cdfs)
    equalized_luminance = interpolate_cdfs(filtered_cdfs, luminance)
    
    # Replace the luminance channel in the original image
    equalized_image = image.copy()
    equalized_image[:, :, 0] = equalized_luminance / 255 * image[:, :, 0]
    equalized_image[:, :, 1] = equalized_luminance / 255 * image[:, :, 1]
    equalized_image[:, :, 2] = equalized_luminance / 255 * image[:, :, 2]

    return equalized_image

#image = cv2.imread('D:\\XuLyAnh\\IMAGE\\aa.jpg')
image = cv2.imread('D:\\XuLyAnh\\IMAGE\\low_quality_img.png')
equalized_image = local_histogram_equalization(image)

cv2.namedWindow('Equalized Image')
cv2.imshow('Equalized Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
