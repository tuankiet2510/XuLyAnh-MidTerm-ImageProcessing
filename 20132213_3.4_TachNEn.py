import cv2
import numpy as np
'''
thiết lập một nền xanh hoặc xanh lá (blue or green screen) và thực hiện tách matte từ hình ảnh. Đây là quá trình tách nền (chroma keying)
 từ một ảnh để loại bỏ nền xanh hoặc xanh lá và có thể thay thế nó bằng một nền khác. Để thực hiện điều này, chúng ta cần hai bức ảnh: một ảnh chỉ có nền mà không có vật thể nào khác và một ảnh với cùng nền đó nhưng có thêm vật thể cần tách.
'''
'''
def chroma_key(background_img_path, object_img_path):
    # Đọc ảnh nền và ảnh với vật thể
    background_img = cv2.imread(background_img_path)
    object_img = cv2.imread(object_img_path)
    height, width = object_img.shape[:2]
    print(height)
    print(width)
    background = cv2.resize(background_img, (852, 480))


    # Chắc chắn rằng cả hai ảnh đều cùng kích thước
    if background.shape != object_img.shape:
        print("Error: Kích thước của hai ảnh không khớp!")
        return None

    # Tính toán sự khác biệt giữa hai ảnh
    diff = cv2.absdiff(background, object_img)

    # Ngưỡng để xác định nền
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Chuyển đổi sang ảnh xám và áp dụng làm mờ để làm mịn mask
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Tạo ảnh alpha (kênh độ trong suốt)
    alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255

    # Tạo ảnh cuối cùng bằng cách sử dụng alpha blending
    foreground = object_img * alpha
    background = background * (1 - alpha)
    result = cv2.add(foreground, background)

    return result
'''
# Đường dẫn đến ảnh nền và ảnh với vật thể
background_path = 'D:\\XuLyAnh\\IMAGE\\green_plain_bg.jpg'
object_path = 'D:\\XuLyAnh\\IMAGE\\elephant_green.png'

# Tạo ảnh kết quả
#result = chroma_key(background_img_path, object_img_path)

# Read the background and object images
background = cv2.imread(background_path)
object_img = cv2.imread(object_path)

# Ensure the background and object image have the same dimensions
if background.shape[:2] != object_img.shape[:2]:
    background = cv2.resize(background, (object_img.shape[1], object_img.shape[0]), interpolation=cv2.INTER_AREA)

# Convert images to float32 for subtraction
background = np.float32(background)
object_img = np.float32(object_img)

# Calculate the absolute difference between the background and the object images
diff = cv2.absdiff(background, object_img)

# Convert the difference to grayscale
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# Threshold the grayscale difference to create a mask
_, mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
# Make sure the mask is a single channel and 8-bit
mask = mask.astype(np.uint8)
# Invert the mask to get the foreground
foreground_mask = cv2.bitwise_not(mask)

# Use the mask to extract the object
extracted_object = cv2.bitwise_and(object_img, object_img, mask=foreground_mask)

# Convert the result to uint8 format
extracted_object = cv2.convertScaleAbs(extracted_object)

# Hiển thị và lưu kết quả nếu thành công
if extracted_object is not None:
    cv2.imshow('Chroma Key Result', extracted_object)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('D:\\XuLyAnh\\IMAGE\\chroma_key_result.jpg', extracted_object)
