import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

def Histogram_equalization() : pass

def CLAHE():pass #(Contrast Limited Adaptive Histogram Equalization) with various tileGridSize from 1 to 1000:

def adjust_contrast(image, contrast):
    #return cv2.addWeighted(image, contrast, image, 0, 128*(1-factor))
    # Alpha controls the contrast (1.0-3.0).
    # Alpha 1.0 means no change.
    # beta control Brightness (0-100)
    #cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return cv2.convertScaleAbs(image, alpha=contrast)
    

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

def adjust_solarize_image(image, threshold): #đảo ngược màu sắc trong một hình ảnh dựa trên một ngưỡng
    # Invert colors based on a threshold value.
    solarized_image = np.where(image < threshold, image, 255 - image) # Hàm np.where kiểm tra từng điểm ảnh trong hình ảnh gốc (image). Nếu giá trị của điểm ảnh nhỏ hơn ngưỡng (threshold), thì giá trị của điểm ảnh trong hình ảnh đảo ngược (solarized_image) sẽ bằng giá trị của điểm ảnh trong hình ảnh gốc (image). Ngược lại, nếu giá trị của điểm ảnh lớn hơn hoặc bằng ngưỡng, thì giá trị của điểm ảnh trong hình ảnh đảo ngược (solarized_image) sẽ bằng 255 - image, tức là màu sắc sẽ bị đảo ngược.
    return solarized_image

#def edge_detection(image):
 #   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  #  edges = cv2.Canny(gray, 100, 200)
   # return edges
def edge_detection(image, low_threshold, high_threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges

def  sharpen_image(image, check):
    if(check == 0):
        return image
    else:
          # Define a sharpening kernel
        sharpening_kernel = np.array([[-1,-1,-1], 
                                    [-1, 9,-1],
                                    [-1,-1,-1]])

        # Apply the sharpening kernel to the image
        sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

        # Return the sharpened image
        return sharpened_image
def apply_sepia(image):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    adjusted = cv2.transform(image, sepia_filter)
    return adjusted

def blur_image(image, k_size):
    adjusted =  cv2.GaussianBlur(image, (k_size, k_size), 0)
    return adjusted

#Contrast and brightness can be adjusted using alpha (α) and beta (β), respectively. These variables are often called the gain and bias parameters. The expression can be written as

def adjust_brightness(image, alpha, beta):
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def adjust_brightnessNcontrast(image,contrast,brightness):
    image = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness) 
    return image

def convert_to_grayscale(image):
    adjusted =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return adjusted

def on_trackbar_change(_):
    """ Callback function for trackbar """
    '''
    global adjusted
    contrast = cv2.getTrackbarPos('Contrast', 'Trackbars')
    brightness = cv2.getTrackbarPos('Brightness', 'Trackbars')
    #adjusted = adjust_contrast(image, contrast)
    solarization = cv2.getTrackbarPos('Solarization', 'Trackbars')
    #adjusted = solarize(adjusted,solarization)
    edge_lower_threshold = cv2.getTrackbarPos('Edge_lower_threshold', 'Trackbars')
    edge_upper_threshold = cv2.getTrackbarPos('Edge_upper_threshold','Trackbars')
    #adjusted = edge_detection(adjusted, edge_lower_threshold, edge_upper_threshold)
    blur_kernel_size = cv2.getTrackbarPos('Blur_kernel_size', 'Trackbars')
    #adjusted = blur_image(adjusted, blur_kernel_size)
    brightness_alpha = cv2.getTrackbarPos('Brightness_Alpha', 'Trackbars') 
    brightness_beta = cv2.getTrackbarPos('Brightness_Beta', 'Trackbars')  # # Điều chỉnh phạm vi từ -50 đến +50
    #adjusted = adjust_brightness(adjusted, brightness_alpha, brightness_beta)
    apply_sepia = cv2.getTrackbarPos('Apply Sepia', 'Trackbars')
    convert_to_grayscale = cv2.getTrackbarPos('Convert to Grayscale', 'Trackbars')
   
    #adjusted = adjust_contrast(image, contrast)
    adjusted = edge_detection(adjusted, edge_lower_threshold, edge_upper_threshold)
    adjusted = blur_image(adjusted, blur_kernel_size)
    adjusted = adjust_brightnessNcontrast(adjusted,contrast,brightness)
   '''
    #adjusted = adjust_brightness(adjusted, brightness_alpha, brightness_beta)   
    '''
    if solarization > 0:
        adjusted = solarize(adjusted, solarization)
    # Sepia và grayscale không nên được áp dụng cùng lúc
    if apply_sepia:
        adjusted = apply_sepia(adjusted)
    elif convert_to_grayscale:
        adjusted = convert_to_grayscale(adjusted)
    '''
    global adjusted, image
    contrast = cv2.getTrackbarPos('Contrast', 'Trackbars')
    brightness = cv2.getTrackbarPos('Brightness', 'Trackbars') - 255
    adjusted = adjust_brightnessNcontrast(image,contrast,brightness)
    solarization = cv2.getTrackbarPos('Solarization', 'Trackbars') 
    adjusted = adjust_solarize_image(adjusted,solarization)
    saturation = cv2.getTrackbarPos('Saturation', 'Trackbars') 
    
    cv2.imshow('Image', adjusted)

def save_image():
    global adjusted
    filepath = filedialog.asksaveasfilename(defaultextension=".jpg")
    if filepath:
        cv2.imwrite(filepath, adjusted)  # Lưu ảnh đã chỉnh sửa
        messagebox.showinfo("Thông báo", "Ảnh đã được lưu thành công!")
        #cv2.putText(adjusted, "Saved!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #cv2.imshow('Image', adjusted)
        #cv2.waitKey(2000)  # Hiển thị thông báo "Saved!" trong 2 giây

def open_image_editor():
    global image, adjusted
    filepath = filedialog.askopenfilename()
    if filepath:
        image = cv2.imread(filepath)
        adjusted = image.copy()
           # Tạo cửa sổ mới cho các thanh trượt
        trackbar_window = "Trackbars"
        cv2.namedWindow(trackbar_window)
        cv2.resizeWindow(trackbar_window, 1200, 600)  # Đặt kích thước cho cửa sổ
          # Thêm các thanh trượt vào cửa sổ này
        
        # Create trackbars for color change and gamma adjustment

        '''
        cv2.createTrackbar('Contrast', trackbar_window, 10, 200, on_trackbar_change)#Tương phản
        cv2.createTrackbar('Solarization', trackbar_window, 2, 255, on_trackbar_change)
        cv2.createTrackbar('Edge_lower_threshold', trackbar_window, 0, 255, on_trackbar_change)
        cv2.createTrackbar('Edge_upper_threshold', trackbar_window, 0, 255, on_trackbar_change)

        cv2.createTrackbar('Blur_kernel_size', trackbar_window, 1, 31, on_trackbar_change)
        cv2.createTrackbar('Brightness_Alpha', trackbar_window, 0, 3, on_trackbar_change) # (điều chỉnh độ sáng) 
        cv2.createTrackbar('Brightness_Beta', trackbar_window, -50, 50, on_trackbar_change) # (điều chỉnh bias).
        cv2.createTrackbar('Apply Sepia',trackbar_window,0 , 1 , on_trackbar_change)
        cv2.createTrackbar('Convert to Grayscale',trackbar_window,0 , 1, on_trackbar_change)
        
        '''
        '''
        #cv2.createTrackbar('Contrast', trackbar_window, 10, 200, on_trackbar_change)#Tương phản
        cv2.createTrackbar('Contrast', trackbar_window, 1,3, on_trackbar_change)#Tương phản 0->3
        cv2.createTrackbar('Brightness', trackbar_window, 0 , 255, on_trackbar_change ) # -255 -> 255 , value = 0
        cv2.createTrackbar('Solarization', trackbar_window, 0, 255, on_trackbar_change)
        cv2.createTrackbar('Edge_lower_threshold', trackbar_window, 0, 255, on_trackbar_change)
        cv2.createTrackbar('Edge_upper_threshold', trackbar_window, 0, 255, on_trackbar_change)

        cv2.createTrackbar('Blur_kernel_size', trackbar_window, 1, 31, on_trackbar_change)
        
        cv2.createTrackbar('Apply Sepia',trackbar_window,0 , 1 , on_trackbar_change)
        cv2.createTrackbar('Convert to Grayscale',trackbar_window,0 , 1, on_trackbar_change)
        '''
        '''
        cv2.createTrackbar('Brightness_Alpha', trackbar_window, 0, 3, on_trackbar_change) # (điều chỉnh độ sáng) 
        cv2.createTrackbar('Brightness_Beta', trackbar_window, -50, 50, on_trackbar_change) # (điều chỉnh bias).
        '''
        cv2.createTrackbar('Contrast', trackbar_window, 1,3, on_trackbar_change)#Tương phản 0->3
        cv2.createTrackbar('Brightness', trackbar_window, 255 , 510, on_trackbar_change ) # 0->510 -255 <-> -255 -> 255 , inital value = 0
        cv2.createTrackbar('Solarization', trackbar_window, find_initial_solarization(adjusted), 255, on_trackbar_change)
        cv2.createTrackbar('Saturation', trackbar_window, 100,200, on_trackbar_change)
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
# Load your image here
#image = cv2.imread('D:\\XuLyAnh\\IMAGE\\Landscape.jpg')
image = None
adjusted = None
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
