import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

def Histogram_equalization() : pass

def CLAHE():pass #(Contrast Limited Adaptive Histogram Equalization) with various tileGridSize from 1 to 1000:
'''
def adjust_contrast(image, contrast):
    #return cv2.addWeighted(image, contrast, image, 0, 128*(1-factor))
    # Alpha controls the contrast (1.0-3.0).
    # Alpha 1.0 means no change.
    # beta control Brightness (0-100)
    #cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return cv2.convertScaleAbs(image, alpha=contrast)
'''   

def solarize(image, levels):
    # Kiểm tra xem ảnh có phải là màu hay không
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Chuyển ảnh màu sang grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #chi hoat dong voi anh mau
    else:
        # Nếu ảnh đã là grayscale, sử dụng trực tiếp
        gray_image = image

    # Quantize the image
    adjusted = np.floor(gray_image / (256 / levels)) * (256 / levels)
    return adjusted

 #Tuong tu  ImageOps.solarize(im1, threshold)  

def adjust_solarize_image(image, threshold,initial_threshold): #đảo ngược màu sắc trong một hình ảnh dựa trên một ngưỡng
    # Invert colors based on a threshold value.
    if threshold == initial_threshold:
        return image
    else:
        solarized_image = np.where(image < threshold, image, 255 - image) # Hàm np.where kiểm tra từng điểm ảnh trong hình ảnh gốc (image). Nếu giá trị của điểm ảnh nhỏ hơn ngưỡng (threshold), thì giá trị của điểm ảnh trong hình ảnh đảo ngược (solarized_image) sẽ bằng giá trị của điểm ảnh trong hình ảnh gốc (image). Ngược lại, nếu giá trị của điểm ảnh lớn hơn hoặc bằng ngưỡng, thì giá trị của điểm ảnh trong hình ảnh đảo ngược (solarized_image) sẽ bằng 255 - image, tức là màu sắc sẽ bị đảo ngược.
        return solarized_image

'''
The saturation_scale parameter typically ranges from 0.0 to 2.0 when adjusting saturation in an image using the HSV color space. Here's what the range means:

0.0: This corresponds to a completely desaturated image where all colors become grayscale. In other words, it removes all color information, leaving only the brightness component.

1.0: This represents the original saturation of the image. No change in saturation occurs when saturation_scale is set to 1.0.

Values between 0.0 and 1.0: These values reduce the saturation of the image, making it less colorful. The closer the value is to 0.0, the less saturated the image becomes.

Values greater than 1.0: These values increase the saturation of the image, making it more colorful. The closer the value is to 2.0, the more saturated the image becomes.
'''
def adjust_saturation(image, saturation_scale): 
    # Convert to HSV and adjust saturation.
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[..., 1] = hsv_image[..., 1] * saturation_scale
    new_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return new_image

'''
The strength parameter in the apply_vignette function controls the intensity or strength of the vignette effect applied to the image. Typically, the strength parameter should have a range between 0.0 and 1.0, where:

0.0: No vignette effect is applied. The image remains unchanged.

1.0: This represents the full strength of the vignette effect, meaning the corners and edges of the image will be significantly darker, creating a strong vignetting effect.

Values between 0.0 and 1.0: These values control the intensity of the vignette effect. The closer the value is to 1.0, the stronger the vignette effect will be. Lower values will result in a less pronounced vignette effect.
'''
def adjust_vignette(image, strength):
    if strength == 0.0: #KO HIỂU SAO NHƯNG STRENGTH = 0 VẪN PHỦ 1 LỚP MASK RẤT ĐẬM
        return image
    # Apply a vignette effect to the image.
    rows, cols = image.shape[:2]
    # Create vignette mask using Gaussian kernels.
    kernel_x = cv2.getGaussianKernel(cols, strength * cols)
    kernel_y = cv2.getGaussianKernel(rows, strength * rows)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    vignette = np.copy(image)
    # Apply the mask to each channel in the image.
    for i in range(3):
        vignette[..., i] = vignette[..., i] * mask
    return vignette


def adjust_blur(image, kernel_size):
    # Apply a blur effect to the image.
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image

def edge_detection(image, low_threshold, high_threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges

def adjust_sepia(image,check):
    if check == 0:
        return image
    else:
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                [0.349, 0.686, 0.168],
                                [0.393, 0.769, 0.189]])
        return cv2.transform(image, sepia_filter)

#Contrast and brightness can be adjusted using alpha (α) and beta (β), respectively. These variables are often called the gain and bias parameters. The expression can be written as

'''
def adjust_brightness(image, alpha, beta):
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted
'''
def adjust_brightnessNcontrast(image,contrast,brightness):
    image = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness) 
    return image

#tăng độ sắc nét của hình ảnh
def adjust_sharpen(image, strength= 0.0, kernel_size = 3):
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
                            [0, 0, 0]])
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
    
def adjust_grayscale(image, check):
    if check == 0:  
        return image
    else:
        adjusted =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return adjusted

'''
def adjust_reduce_noise(image,check):
    # Reduce noise in the image.
    if check == 0:
        return image
    else:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
'''
def adjust_reduce_noise(image,check, h=20, hForColorComponents=10, templateWindowSize=7, searchWindowSize=21):
    # Reduce noise in the image.
    if check == 0:
        return image
    else:
        denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h, hForColorComponents, templateWindowSize, searchWindowSize)
        return denoised_image

def create_exposure_series(image, contrast_range, brightness_range, num_images):
    steps = len(contrast_range)
    images = []
    #for i in range(steps):
    for i in range(num_images):
        adjusted_image = adjust_brightnessNcontrast(image, contrast_range[i], brightness_range[i])
        images.append(adjusted_image)
    
    return images
'''
def adjust_HDR(image,check):
    # Tạo các phiên bản có độ sáng và tương phản khác nhau
    if(check == 0):
        return image
    else:

        bright_img = adjust_brightnessNcontrast(image, 1.5, 50)
        dark_img = adjust_brightnessNcontrast(image, 0.5, -50)

        # Tạo một danh sách các ảnh để xử lý HDR
        images = [dark_img, image, bright_img]

       
        # Giả lập danh sách các thời gian phơi sáng cho mỗi ảnh
        # Ví dụ: [1/30, 1/15, 1/8] có thể là các giá trị phơi sáng giả lập
        # Điều này không chính xác nhưng cần thiết cho quy trình xử lý HDR
        times = np.array([1/30, 1/15, 1/8], dtype=np.float32)

        # Kết hợp các ảnh
        hdr = cv2.createMergeDebevec().process(images, times)
        # Áp dụng tonemapping để chuyển đổi HDR sang LDR
        tonemap = cv2.createTonemap(2.2)
        ldr = tonemap.process(hdr)

        # Chuyển đổi kết quả sang 8-bit và trả về
        hdr_8bit = cv2.normalize(ldr, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        return hdr_8bit
'''

def adjust_HDR(image, num_images, check):
    if(check == 0):
        return image
    else:
        if num_images == 1:
            return image
        else:
            # Định nghĩa các bước điều chỉnh tương phản và độ sáng
            contrast_range = np.linspace(0.5, 1.5, num=num_images)
            brightness_range = np.linspace(-50, 50, num=num_images)
            
            # Tạo series ảnh với các mức độ phơi sáng khác nhau
            images = create_exposure_series(image, contrast_range, brightness_range, num_images)
            print(len(images))
            # Giả lập danh sách thời gian phơi sáng dựa trên số lượng ảnh
            times = np.array([1/(2**i) for i in range(num_images)], dtype=np.float32)

            #calibrate = cv2.createCalibrateDebevec()
            #response = calibrate.process(images, times)
            # Kết hợp các ảnh thành HDR
            
            merge_debevec = cv2.createMergeDebevec()

            #hdr = merge_debevec.process(images, times,response)
            hdr = merge_debevec.process(images, times)
            # Áp dụng tonemapping
            tonemap = cv2.createTonemap(2.2)
            ldr = tonemap.process(hdr)

            # Chuyển đổi kết quả sang 8-bit và trả về
            hdr_8bit = cv2.normalize(ldr, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            #merge_mertens = cv2.createMergeMertens()
            return hdr_8bit

def on_trackbar_change(_):
   
    #adjusted = adjust_brightness(adjusted, brightness_alpha, brightness_beta)   
    global adjusted,image
    
    contrast = cv2.getTrackbarPos('Contrast', 'Trackbars') / 100
    brightness = cv2.getTrackbarPos('Brightness', 'Trackbars') - 255
    adjusted = adjust_brightnessNcontrast(image,contrast,brightness)
    
    solarization = cv2.getTrackbarPos('Solarization', 'Trackbars') 
    adjusted = adjust_solarize_image(adjusted,solarization,initial_solarization)
    saturation = cv2.getTrackbarPos('Saturation', 'Trackbars') / 100 
    
    adjusted = adjust_saturation(adjusted,saturation)
    vignette_strength = cv2.getTrackbarPos('Vignette', 'Trackbars') / 1000
    adjusted = adjust_vignette(adjusted,vignette_strength)
    blur_kernel_size = cv2.getTrackbarPos('Blur','Trackbars')
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1

    adjusted = adjust_blur(adjusted,blur_kernel_size)
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
    cv2.imshow('Image', adjusted)

def save_image():
    global adjusted
    filepath = filedialog.asksaveasfilename(defaultextension=".jpg")
    if filepath:
        cv2.imwrite(filepath, adjusted)  # Lưu ảnh đã chỉnh sửa
        messagebox.showinfo("Thông báo", "Ảnh đã được lưu thành công!")

def open_image_editor():
    global image, adjusted , initial_solarization
    filepath = filedialog.askopenfilename()
    if filepath:
        image = cv2.imread(filepath)
        adjusted = image.copy()
        initial_solarization = find_initial_solarization1(adjusted)
           # Tạo cửa sổ mới cho các thanh trượt
        trackbar_window = "Trackbars"
        cv2.namedWindow(trackbar_window)
        cv2.resizeWindow(trackbar_window, 1600, 600)  # Đặt kích thước cho cửa sổ
          # Thêm các thanh trượt vào cửa sổ này
        
        # Create trackbars for color change and gamma adjustment
        
        cv2.createTrackbar('Contrast', trackbar_window, 100,300, on_trackbar_change)#Tương phản 0->3
        cv2.createTrackbar('Brightness', trackbar_window, 255 , 510, on_trackbar_change ) # 0->510 -255 <-> -255 -> 255 , inital value = 0
        
        cv2.createTrackbar('Solarization', trackbar_window, initial_solarization, 255, on_trackbar_change) # Base là 120
        cv2.createTrackbar('Saturation', trackbar_window, 100,200, on_trackbar_change) # 0 ->2 , bằng 1 thì giữ nguyên
        cv2.createTrackbar('Vignette', trackbar_window, 0 , 1000, on_trackbar_change )
        cv2.createTrackbar('Blur', trackbar_window, 1, 51, on_trackbar_change)
        cv2.createTrackbar('Sepia',trackbar_window, 0 , 1 , on_trackbar_change)
        cv2.createTrackbar('ReduceNoise',trackbar_window, 0 , 1 , on_trackbar_change)
        cv2.createTrackbar('GrayScale',trackbar_window, 0 , 1 , on_trackbar_change)
        #cv2.createTrackbar('Sharpen',trackbar_window, 0 , 1 , on_trackbar_change)
        cv2.createTrackbar('Sharpen_Strength',trackbar_window, 0 , 31 , on_trackbar_change)
        cv2.createTrackbar('Sharpen_KernelSize',trackbar_window, 0, 4 , on_trackbar_change)
        cv2.createTrackbar('ApplyHDR',trackbar_window, 0 , 1 , on_trackbar_change)
        # Show the image
        cv2.namedWindow("Image")
        on_trackbar_change(0)  # Initial call to display the 
        cv2.waitKey(0) # Wait for a key press and then terminate the program
        cv2.destroyAllWindows()

def find_initial_solarization(image):
    # Tính giá trị solarization ban đầu của ảnh
    # Bạn có thể thực hiện tính toán dựa trên hình ảnh để tìm giá trị này
    # Ví dụ: Tính trung bình của tất cả các giá trị điểm ảnh trong ảnh
    # và sử dụng nó làm giá trị solarization ban đầu.
    initial_solarization = int(np.mean(image))
    return initial_solarization
def find_initial_solarization1(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Tính histogram của ảnh grayscale
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Tìm giá trị trung bình của histogram
    total = 0
    count = 0
    for i in range(256):
        total += i * hist[i]
        count += hist[i]

    if count == 0:
        return 128  # Trả về một giá trị mặc định nếu không có điểm ảnh nào

    initial_solarization = int(total / count)
    return initial_solarization
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
