import cv2 as cv
import numpy as np
#Làm theo tutorial cua opencv https://docs.opencv.org/3.4/d4/d1b/tutorial_histogram_equalization.html
# Function to limit the local gain
def limit_gain(hist, clip_limit):
    excess = 0
    for i in range(len(hist)):
        excess += max(hist[i] - clip_limit, 0)
        hist[i] = min(hist[i], clip_limit)
    excess_per_bin = excess // len(hist)
    for i in range(len(hist)):
        hist[i] += excess_per_bin
    return hist

# Function to adjust histogram for black and white mapping
def adjust_histogram(hist, image_size, black_white_ratio):
    clip_limit = image_size * black_white_ratio
    return limit_gain(hist, clip_limit)

# Main function
def histogram_equalization(input_image_path):
    # Load the image
    
    src = cv.imread(input_image_path)
    if src is None:
        print(f'Could not open or find the image: {input_image_path}')
        exit(0)

    # Convert to grayscale
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    
    dst = cv.equalizeHist(gray)
    '''
    # Compute the histogram
    hist = cv.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.ravel()

    # Optional: Adjust histogram for black and white
    adjusted_hist = adjust_histogram(hist, gray.size, 0.05)  # 5% pixels to black and white

    # Cumulative distribution and equalization
    cum_hist = np.cumsum(adjusted_hist)
    cum_hist_normalized = cum_hist * 255 / cum_hist[-1]
    equalized = cv.LUT(gray, cum_hist_normalized.astype('uint8'))

    # Re-generate the color image (optional step)
    # ... (This part depends on specific requirements for color reintegration)
    '''
    # Display results
    cv.imshow('Source Image', src)
    cv.imshow('Equalized Image', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Run the function
#image = cv2.imread('D:\\XuLyAnh\\IMAGE\\moon.png')
histogram_equalization('D:\\XuLyAnh\\IMAGE\\moon.png')
#histogram_equalization('D:\\XuLyAnh\\IMAGE\\history_src_img.jpg')
