import cv2
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import messagebox

#KO CẦN DÙNG biến check đẻ return ảnh gốc , khi kernel_size=1 thì ảnh như cũ
def adjust_gaussian_blur(image, sigma, kernel_size = 1):
    # Apply a blur effect to the image.
    '''
    if(check == 0):
        return image
    else:
        adjusted = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma )
        return adjusted
    '''
    
    adjusted = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma )
    return adjusted

def adjust_median_blur(image, kernel_size = 1):
    '''
    if(check == 0):
        return image
    else:
        adjusted = cv2.medianBlur(image, kernel_size)
        return adjusted
    '''
    adjusted = cv2.medianBlur(image, kernel_size)
    return adjusted

def adjust_bilateral_filter(image, diameter, sigmaColor, sigmaSpace):
    if diameter == 1:
        return image
    else:
        adjusted = cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)
        return adjusted

def adjust_sharpen(image, strength= 0.0, kernel_size = 1):
    '''
    if(check == 0):
        return image
    '''
        # Generate a sharpening kernel
    # The strength factor controls the intensity of the sharpening effect
    '''
    kernel = np.array([[-1 * strength, -1 * strength, -1 * strength], 
                       [-1 * strength, 9 * strength + 1, -1 * strength],
                       [-1 * strength, -1 * strength, -1 * strength]])
    '''
    if(strength < 0):
            kernel = np.array([[0, 0, 0], 
                            [0, 1, 0],
                            [0, 0, 0]]) #Anhr goc
    else:
        kernel = np.array([[-1, -1, -1], 
                        [-1, 9 + strength, -1],
                        [-1, -1, -1]])
    
    if kernel_size > 3:
        kernel = cv2.resize(kernel, (kernel_size, kernel_size), interpolation=cv2.INTER_LINEAR)

    # Apply the sharpening kernel to the image
    sharpened_image = cv2.filter2D(image, -1, kernel)

    # Return the sharpened image
    return sharpened_image

#Contrast and brightness can be adjusted using alpha (α) and beta (β), respectively. These variables are often called the gain and bias parameters. The expression can be written as

'''
def adjust_brightness(image, alpha, beta):
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted
'''

def adjust_reduce_noise(image,check, h=20, hForColorComponents=10, templateWindowSize=7, searchWindowSize=21):
    # Reduce noise in the image.
    if check == 0:
        return image
    else:
        denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h, hForColorComponents, templateWindowSize, searchWindowSize)
        return denoised_image

def on_trackbar_change(_):
   
    #adjusted = adjust_brightness(adjusted, brightness_alpha, brightness_beta)   
    '''
    global adjusted,image
    
    kernel_size = cv2.getTrackbarPos('KernelSz','Trackbars')
    if kernel_size % 2 == 0:
        kernel_size += 1

    adjusted = adjust_gaussian_blur(image,kernel_size)
    #print( vignette_strength )

    apply_sepia = cv2.getTrackbarPos('Sepia', 'Trackbars')
    adjusted = adjust_sepia(adjusted,apply_sepia)

    apply_reduce_noise = cv2.getTrackbarPos('ReduceNoise', 'Trackbars') 
    adjusted = adjust_reduce_noise(adjusted,apply_reduce_noise)

    apply_grayscale = cv2.getTrackbarPos('GrayScale','Trackbars')
    adjusted = adjust_grayscale(adjusted,apply_grayscale)

    #apply_sharpen = cv2.getTrackbarPos('Sharpen','Trackbars')
    #adjusted = adjust_sharpen(adjusted,apply_sharpen)

    sharpen_strength = (cv2.getTrackbarPos('Sharpen_Strength', 'Trackbars') -1) /10
    sharpen_kernel_size = cv2.getTrackbarPos('Sharpen_KernelSize', 'Trackbars') + 3
    adjusted = adjust_sharpen(adjusted,sharpen_strength,sharpen_kernel_size)

    apply_HDR =   cv2.getTrackbarPos('ApplyHDR', 'Trackbars') 
    adjusted = adjust_HDR(adjusted,12,apply_HDR)
    #cv2.imshow('Image', adjusted)
    '''
    global image, adjusted
    kernel_size = cv2.getTrackbarPos('KernelSz','Trackbars')
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigmaGau = cv2.getTrackbarPos('SigmaGau','Trackbars')
    gaussian_blur = adjust_gaussian_blur(adjusted,sigmaGau,kernel_size)

    median_blur = adjust_median_blur(adjusted,kernel_size)

    diameter = cv2.getTrackbarPos('Diameter','Trackbars')
    sigmaColor = cv2.getTrackbarPos('SigmaColor','Trackbars')
    sigmaSpace =  cv2.getTrackbarPos('SigmaSpace','Trackbars')
    bilateral_filter = adjust_bilateral_filter(adjusted,diameter,sigmaColor,sigmaSpace)

    sharpen_strength = (cv2.getTrackbarPos('SharpenStrenght', 'Trackbars') -1) /10
    sharpen_kernel_size = cv2.getTrackbarPos('Sharpen_KernelSize', 'Trackbars') + 3

    #sharpened_image = adjust_sharpen(adjusted,sharpen_strength,kernel_size)
    sharpened_image = adjust_sharpen(adjusted,sharpen_strength,sharpen_kernel_size)


    titles = ['Original Image', 'Gaussian Blurring', 'Median Blurring', 'Bilateral Filter', 'Sharpened Image']
    images = [image, gaussian_blur, median_blur, bilateral_filter,image]
    
    #plt.figure(figsize=(16, 8))
    '''
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])  # Hide tick marks
    plt.show()
    '''
    plt.subplot(2, 3, 1)  # Có 5 hàng, 1 cột, subplot ở vị trí 1
    plt.imshow(image, cmap='gray')
    plt.title('Image ')

    plt.subplot(2, 3, 2)  # Có 5 hàng, 1 cột, subplot ở vị trí 2
    plt.imshow(gaussian_blur, cmap='gray')
    plt.title('Gauss blur')

    plt.subplot(2, 3, 3)  # Có 5 hàng, 1 cột, subplot ở vị trí 1
    plt.imshow(median_blur, cmap='gray')
    plt.title('Median blur')

    plt.subplot(2, 3, 4)  # Có 5 hàng, 1 cột, subplot ở vị trí 2
    plt.imshow(bilateral_filter, cmap='gray')
    plt.title('bilateral')

    plt.subplot(2, 3, 5)  # Có 5 hàng, 1 cột, subplot ở vị trí 2
    plt.imshow(sharpened_image, cmap='gray')
    plt.title('sharpen')
    plt.show()

def save_image():
    global adjusted
    filepath = filedialog.asksaveasfilename(defaultextension=".jpg")
    if filepath:
        cv2.imwrite(filepath, adjusted)  # Lưu ảnh đã chỉnh sửa
        messagebox.showinfo("Thông báo", "Ảnh đã được lưu thành công!")

def open_image_editor():
    global image, adjusted 
    filepath = filedialog.askopenfilename()
    if filepath:
        image1 = cv2.imread(filepath)
        image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        adjusted = image.copy()
    
           # Tạo cửa sổ mới cho các thanh trượt
        trackbar_window = "Trackbars"
        cv2.namedWindow(trackbar_window)
        cv2.resizeWindow(trackbar_window, 1600, 600)  # Đặt kích thước cho cửa sổ
          # Thêm các thanh trượt vào cửa sổ này
        
        # Create trackbars for color change and gamma adjustment
        
        cv2.createTrackbar('KernelSz', trackbar_window, 1,51    , on_trackbar_change)#Tương phản 0->3
        cv2.createTrackbar('SigmaGau', trackbar_window, 0, 100, on_trackbar_change ) # =0 thì sẽ tự động tính toán ,  = 1 thì 
        
        cv2.createTrackbar('Diameter', trackbar_window,  1,255, on_trackbar_change) # Base là 120
        cv2.createTrackbar('SigmaColor', trackbar_window, 0,400, on_trackbar_change) # 0 ->2 , bằng 1 thì giữ nguyên
        cv2.createTrackbar('SigmaSpace', trackbar_window, 0, 400, on_trackbar_change )
        cv2.createTrackbar('SharpenStrenght', trackbar_window, 0 , 31, on_trackbar_change )
        cv2.createTrackbar('Sharpen_KernelSize',trackbar_window, 0, 4 , on_trackbar_change)

        # Show the image
        #cv2.namedWindow("Image")
        plt.figure(figsize=(16, 10))
        on_trackbar_change(0)  # Initial call to display the 
        cv2.waitKey(0) # Wait for a key press and then terminate the program
        cv2.destroyAllWindows()


# Load your image here
#image = cv2.imread('D:\\XuLyAnh\\IMAGE\\Landscape.jpg')
image = None
adjusted = None
initial_solarization = None
# Create a window
root = tk.Tk()
root.geometry("400x600")
# Calculate the position for the center of the screen
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - 400) / 2
y = (screen_height - 600) / 2

# Set the root window position to the center
root.geometry("+%d+%d" % (x, y))

#create open file dialog
open_button = tk.Button(root, text="Open image file", command=open_image_editor)
open_button.pack()

save_button = tk.Button(root, text="Save adjusted image",command=save_image)
save_button.pack()
root.mainloop()
