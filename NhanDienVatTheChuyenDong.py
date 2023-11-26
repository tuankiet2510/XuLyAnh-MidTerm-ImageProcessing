import cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)

#Lay bg frame
for i in range(10):
    _, frame = cap.read()
frame = cv2.resize(frame, (640,480))
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
last_frame = gray

while True:
    _, frame = cap.read()
    #Xu ly frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(25,25),0) # lọc nhiễu
    abs_img = cv2.absdiff(last_frame, gray)
    # 0 - 1 =  tràn số -> Nhiễu ảnh nên ph dung abs(0-1) =1 
    last_frame = gray #gán bước ảnh htai cho last_frame

    _, img_mask = cv2.threshold(abs_img,30,255,cv2.THRESH_BINARY)
    contours , _ = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # trả về 1 list các contour

    for contour in contours:
        if(cv2.contourArea(contour) < 900):
            continue #bỏ qua contour có kích thước < 900

        x , y , w ,h = cv2.boundingRect(contour) #Láy ra w,h 
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3) # vẽ hcn
    
    #cv2.imshow('Window', img_mask)
    cv2.imshow('Window', frame)
    if cv2.waitKey(1) == ord('q'):
        break
'''
_, frame = cap.read()
frame = cv2.resize(frame,(640,480))
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
background = gray

cv2.imshow('Background', background)
while True:
    _, frame = cap.read()
    cv2.imshow('Window', frame)
    if cv2.waitKey(1) == ord('q'):
        break
'''