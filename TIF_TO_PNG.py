import os
from PIL import Image
import numpy as np

# Đường dẫn đến thư mục chứa các file 16-bit integer
folder_path = r'D:\XuLyAnh\IMAGE\cthead-16bitABC'

# Tạo folder mới tên là 'PNG' nếu nó chưa tồn tại
png_folder_path = os.path.join(folder_path, 'PNG')
if not os.path.exists(png_folder_path):
    os.makedirs(png_folder_path)

# Lặp qua tất cả các file trong thư mục
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.tif', '.tiff')):
        # Đường dẫn đầy đủ tới file TIF
        tif_file_path = os.path.join(folder_path, filename)

        # Đọc file TIF
        image = Image.open(tif_file_path)

        # Chuyển đổi ảnh sang mảng NumPy
        img_array = np.array(image)

        # Tìm giá trị min và max của ảnh
        min_val = np.min(img_array)
        max_val = np.max(img_array)

        # Chia tỷ lệ giá trị pixel từ 16-bit xuống 8-bit
        img_array = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        # Tạo ảnh mới từ mảng NumPy
        new_image = Image.fromarray(img_array)

        # Lấy tên file mà không bao gồm phần mở rộng
        image_name = os.path.splitext(filename)[0]

        # Tạo đường dẫn của file ảnh đích trong folder 'PNG'
        image_dest_path = os.path.join(png_folder_path, f'{image_name}.png')

        # Lưu file ảnh đích dưới dạng PNG
        new_image.save(image_dest_path, 'PNG')

        # Đóng file TIF
        image.close()

print("Chuyển đổi thành công tất cả ảnh TIF sang PNG.")
