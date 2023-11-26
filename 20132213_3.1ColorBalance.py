import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
def adjust_rgb_color_balance(image, red_balance, green_balance, blue_balance):
    adjusted_img = image.copy()
    #dùng np.clip() đảm bảo giá trị màu sau điều chỉnh nằm trong range(0-255)
    adjusted_img[:,:, 0] = np.clip(image[:,:,0] * blue_balance,0,255)
    adjusted_img[:,:, 1] = np.clip(image[:,:,1] * green_balance,0,255)
    adjusted_img[:,:, 2] = np.clip(image[:,:,2] * red_balance,0,255)
    return adjusted_img
                                   
def adjust_gamma(image, gamma):
    """ Adjust the gamma of an image """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def on_trackbar_change(_):
    """ Callback function for trackbar """
    global adjusted
    red = cv2.getTrackbarPos('Red', 'Image') / 100
    green = cv2.getTrackbarPos('Green', 'Image') / 100
    blue = cv2.getTrackbarPos('Blue', 'Image') / 100
    gamma = cv2.getTrackbarPos('Gamma', 'Image') / 100
    adjusted = adjust_rgb_color_balance(image, red, green, blue)
    adjusted = adjust_gamma(adjusted, gamma)
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
        cv2.namedWindow("Image")
        # Create trackbars for color change and gamma adjustment
        cv2.createTrackbar('Red', 'Image', 100, 200, on_trackbar_change)
        cv2.createTrackbar('Green', 'Image', 100, 200, on_trackbar_change)
        cv2.createTrackbar('Blue', 'Image', 100, 200, on_trackbar_change)
        cv2.createTrackbar('Gamma', 'Image', 100, 300, on_trackbar_change)
        #cv2.createButton('Save Image',save_image)
        # Show the image
        on_trackbar_change(0)  # Initial call to display the 
        cv2.waitKey(0) # Wait for a key press and then terminate the program
        cv2.destroyAllWindows()


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

#root.withdraw() #Hide main window
#create open file dialog
open_button = tk.Button(root, text="Open image file", command=open_image_editor)
open_button.pack()

save_button = tk.Button(root, text="Save adjusted image",command=save_image)
save_button.pack()
root.mainloop()







