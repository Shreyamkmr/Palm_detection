import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
    blur= cv.GaussianBlur(gray, (9,9), 0)
    canny = cv.Canny(blur,60,100)
    return canny

cap = cv.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([0, 10, 60], dtype = "uint8")
    upper_blue = np.array([20, 150, 255], dtype = "uint8")
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    canny_img = canny(res)
    

    #cv.imshow('mask',mask)
    #cv.imshow('res',res)
    #cv.imshow('canny',canny_img)
    kernel = np.ones((3,3),np.uint8)
    kernel_2 = np.ones((2,2),np.uint8)
   # kernel_smooth = np.ones((5,5),np.float32)/25
    
    dialation = cv.dilate(canny_img,kernel ,iterations = 9)
    erosion = cv.erode(dialation,kernel_2,iterations = 10)
   # dst = cv2.filter2D(erosion,-1,kernel)
    blur = cv.GaussianBlur(erosion,(5,5),0)
    #ret, thresh = cv.threshold(blur, 127, 255, 0)

    _, contours,_ = cv.findContours(blur, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    for cnt in contours:
        rect = cv.minAreaRect(cnt)   
        width =rect[1][0]
        heiht= rect[1][1]
    cv.imshow('cont_img',erosion)
    cv.imshow('frame',frame)
    cv.convexityDefects(contours,convexhull,convexityDefects_a)

   
    contour_img= cv.drawContours(frame, [[45,50],[40,60]], -1, (0,255,0), 2)




    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()