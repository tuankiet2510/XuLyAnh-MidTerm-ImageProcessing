import numpy as np
import cv2

# Load the image
image_path = 'D:\\XuLyAnh\\IMAGE\\low_quality_img.png'
image = cv2.imread(image_path)

# Function to compute the luminance of the image
def compute_luminance(image):
    return 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]

# Function to calculate an optimal patch size for the image
def calculate_patch_size(image, max_size=64, min_size=8):
    rows, cols = image.shape[:2]
    return max(min_size, min(max_size, rows // 10, cols // 10))

# Function to compute histograms for each patch of the image
def compute_histograms(image, patch_size):
    rows, cols = image.shape[:2]
    histograms = np.zeros((rows // patch_size, cols // patch_size, 256))
    for i in range(0, rows - patch_size + 1, patch_size):
        for j in range(0, cols - patch_size + 1, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            hist = cv2.calcHist([patch], [0], None, [256], [0, 256])
            histograms[i // patch_size, j // patch_size, :] = hist.flatten()
    return histograms

# Function to distribute the histogram values to adjacent vertices using bilinear interpolation
def distribute_values(histograms, patch_size):
    rows, cols, _ = histograms.shape
    values = np.zeros((rows * patch_size, cols * patch_size))
    for i in range(rows):
        for j in range(cols):
            values[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = histograms[i, j].sum()
    return values

# Function to convert histograms at each vertex to cumulative distribution functions (CDFs)
def convert_to_cdf(values):
    return values.cumsum(axis=0).cumsum(axis=1)

# Function to apply the lookup table to the original luminance values of the image
def apply_lookup(image, lookup):
    rows, cols = image.shape[:2]
    result = np.zeros_like(image, dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            for k in range(3):
                pixel_value = int(image[i, j, k])
                result[i, j, k] = lookup[min(pixel_value, 255)]
    return result

# Function to perform local histogram equalization on an image
def local_histogram_equalization(image):
    luminance = compute_luminance(image).astype(np.uint8)
    patch_size = calculate_patch_size(image)
    histograms = compute_histograms(luminance, patch_size)
    distributed_values = distribute_values(histograms, patch_size)
    cdf = convert_to_cdf(distributed_values)
    lookup = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    lookup = lookup.astype(np.uint8)
    return apply_lookup(image, lookup)

# Applying local histogram equalization to the loaded image
equalized_image = local_histogram_equalization(image)

# Display the original and equalized image using OpenCV
cv2.imshow('Original Image', image)
cv2.imshow('Equalized Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
