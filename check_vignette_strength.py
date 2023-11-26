import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import sys

def adjust_vignette(image, strength):
    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, strength * cols)
    kernel_y = cv2.getGaussianKernel(rows, strength * rows)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    vignette = np.copy(image)
    for i in range(3):
        vignette[..., i] = vignette[..., i] * mask
    return vignette

def calculate_ssim(image1, image2):
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    return ssim(image1_gray, image2_gray, data_range=image2_gray.max() - image2_gray.min())

def main(image_path):
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Không thể đọc ảnh. Vui lòng kiểm tra lại đường dẫn.")
        return

    max_strength = 1.0
    step = 0.01
    threshold_ssim = 0.99
    last_strength_below_threshold = 0

    for strength in np.arange(0, max_strength, step):
        vignetted_image = adjust_vignette(original_image, strength)
        similarity = calculate_ssim(original_image, vignetted_image)
        if similarity < threshold_ssim:
            break
        last_strength_below_threshold = strength

    print(f"Giá trị strength mà tại đó ảnh không thay đổi đáng kể: {last_strength_below_threshold}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Cách sử dụng: python script.py <đường dẫn ảnh>")
    else:
        main(sys.argv[1])
