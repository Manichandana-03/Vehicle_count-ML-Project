#I'm doing this project in pycharm
#1.You need to install cv2 and numpy
#go to file and then settings
#choose python Interpreter
#click on '+' symbol and type opencv
#click on install,do the same for numpy
#then importing cv2 and numpy

import cv2
import numpy as np

#web camera
#capturing the video
cap = cv2.VideoCapture('video.mp4')

#Initialize Subtractor
#cv2.bgsegm.createBackgroundSubtractorMOG() -  Is a algorithm
#which is used to substracts background
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

min_width_rect = 80
min_height_rect = 80
count_line_position = 550


def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

detect = []

#Allowable error between pixel
offset = 6

counter = 0


while True:
    # ret - is a boolean variable that returns true if the frame is available.
    # frame1 - is an image array vector captured based on the default frames per second defined explicitly or implicitly
    ret,frame1 = cap.read()

    #converting image to gray color
    gray =cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    #Image Smoothing using Gaussian Blur
    #-----cv2.GaussianBlur(img,ksize,sigmaX,dst,sigmaY,borderType)--
    blur = cv2.GaussianBlur(gray,(3,3),5)

    #applying on each frame
    img_sub = algo.apply(blur)

    #dilate is morphological operation used to accentuate features
    #dilate increases tlhe object area
    #------cv2.dilate(img,kernel,iterations)---
    dilat = cv2.dilate(img_sub,np.ones((5,5)))

    #cv2.getStructuringElement() -> is use elliptical/circular shaped kernel
    #cv2.getStructuringElement(shape,size) of kernel
    #With the help of kernel we blur an image
    #kernel ->changes the value of given pixel by combining it with different amounts of neighbouring pixels.
    # Then is ti applied to every pixel in the image one-by-one to produce the final image(convolution)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    #Using morphologyEx ->applying dilation followed by erosion
    #It removes small black points on object
    dilatada = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatada = cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernel)

    #contours -> used to join all continuous points have same intensity or color
    #Here h is hierarchy
    #---------------cv2.findContours(source_img,contour_retrevial_mode,contour_approximation_method)-----
    counterShape,h = cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #cv2.line() -> is used a draw a line on the image
    #cv2.line(img,start_point,end_point,color,thickness)
    #start and end points are x,y coordinates
    cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,0,0),3)

    #enumerate() -> adds a counter to an iterabel and returns.
    for (i,c) in enumerate(counterShape):
        #cv2.boundingRect -> used to draw an appropriate rectangle aroung the binary image
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>=min_width_rect) and (h>=min_height_rect)
        if not validate_counter:
            continue

        #cv2.rectangle(img,st_pt,end_pt,color,thickness) -> used to draw a rectangle on the image
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)

        #cv2.putText() -> is used to draw text string on the image
        #cv2.putText(img,text,org,font,fontScale,color[,thickness[,lineType[,bottomLeftOrigin]]])
        cv2.putText(frame1,'Vehicle'+str(counter),(x,y-20),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,0),2)

        center = center_handle(x,y,w,h)

        #appending into detect list
        detect.append(center)

        #cv2.circle() -> is used to draw circle on the image
        #cv2.circle(img,center_coordinates,radius,color,thickness)
        cv2.circle(frame1,center,4,(0,0,255),-1)

        for (x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter +=1
                cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,255,0),3)
                detect.remove((x,y))
                print('Vehicle Counter:'+str(counter))
    cv2.putText(frame1,"VEHICLE COUNTER:"+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)

    #cv2.imshow('Detecter',dilatada)
    cv2.imshow('Video Original',frame1)

    if cv2.waitKey(1) == 13:
        break
cv2.destroyAllWindows()

#releasing the video
cap.release()