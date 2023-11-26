import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def apply_separable_convolution(image, kernel_horizontal, kernel_vertical, padding_type='zero'):
    """
    Apply separable convolution to an image using horizontal and vertical kernels.
    
    :param image: 2D numpy array (grayscale) or 3D numpy array (color)
    :param kernel_horizontal: 1D numpy array, the horizontal kernel
    :param kernel_vertical: 1D numpy array, the vertical kernel
    :param padding_type: String, type of padding ('zero', 'replicate', etc.)
    :return: Convolved image
    """
    # Reshape kernels to 2D
    kernel_horizontal = kernel_horizontal.reshape(1, -1)
    kernel_vertical = kernel_vertical.reshape(-1, 1)

    # Adjust padding type for scipy's convolve2d
    if padding_type == 'zero':
        pad_mode = 'fill'      # Zero padding
    elif padding_type == 'replicate':
        pad_mode = 'symm'      # Symmetric padding

    # Apply convolution for each channel separately if it's a color image
    if len(image.shape) == 3:
        convolved_image = np.zeros_like(image)
        for channel in range(image.shape[2]):
            # Convolve each channel separately
            convolved_channel = convolve2d(image[:,:,channel], kernel_horizontal, mode='same', boundary=pad_mode, fillvalue=0)
            convolved_channel = convolve2d(convolved_channel, kernel_vertical, mode='same', boundary=pad_mode, fillvalue=0)
            convolved_image[:,:,channel] = convolved_channel
    else:
        # Grayscale image
        convolved_image = convolve2d(image, kernel_horizontal, mode='same', boundary=pad_mode, fillvalue=0)
        convolved_image = convolve2d(convolved_image, kernel_vertical, mode='same', boundary=pad_mode, fillvalue=0)

    return convolved_image

def process_image(image_path, kernel_horizontal, kernel_vertical, padding_type='zero', save_path=None):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    '''
    cv2.imshow('image',image)
    cv2.waitKey(0)
    '''
    
    # Check if image is loaded properly
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Apply separable convolution
    convolved_image = apply_separable_convolution(image_rgb, kernel_horizontal, kernel_vertical, padding_type)
    #cv2.imshow('Image',convolved_image)
    #cv2.imshow('applied')
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    #Open cv đọc ảnh theo định dạng BGR
    #Matplotliv hiển thị theo định dạng RGB
    plt.imshow(image_rgb)
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(convolved_image)
    plt.title(' Separable filter')
    
    plt.show()
    # Save or return the image
    '''
    if save_path:
        cv2.imwrite(save_path, convolved_image)
        print(f"Processed image saved at {save_path}")
    else:
        return convolved_image
    '''

# Define kernels (example kernels, replace with actual kernels)
kernel_horizontal = np.array([1, 0, -1])  # Example horizontal kernel
kernel_vertical = np.array([1, 2, 1])     # Example vertical kernel

# Process an image
#input_image_path = 'D:\\XuLyAnh\\IMAGE\\moon.png'  # Replace with actual image path
#input_image_path = 'D:\\XuLyAnh\\IMAGE\\aa.jpg' 
#input_image_path = 'D:\\XuLyAnh\\IMAGE\\transparent-apple-21.png' 
#input_image_path = 'D:\\XuLyAnh\\IMAGE\\Hand_Puppet.jpg'
#input_image_path = 'D:\\XuLyAnh\\IMAGE\\2puppet.jpg'
input_image_path = 'D:\\XuLyAnh\\IMAGE\\2ppl.png'
output_image_path = 'D:\\XuLyAnh\\IMAGE\\moon1.png'  # Replace with desired output path

# Call the function
process_image(input_image_path, kernel_horizontal, kernel_vertical, padding_type='zero', save_path=output_image_path)
