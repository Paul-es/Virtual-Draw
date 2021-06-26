import cv2
import numpy as np
 
cap = cv2.VideoCapture(0)
noiseth = 800
canvas = None

def find():
    flag=0
    while(1):
        _, frame = cap.read()
        cv2.imshow('frame', frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_list=[]
        
        l=[[[55, 130, 92],[90, 255, 255]],[[0, 157, 197],[20, 255, 255]]]
        Dict = {0: 'green', 1: 'red'}
        for i in range(0,len(l)):
            lower_color=np.array(l[i][0], np.uint8)
            upper_color=np.array(l[i][1], np.uint8)
            mask = cv2.inRange(hsv, lower_color, upper_color)

            res = cv2.bitwise_and(frame,frame, mask= mask)

            edges = cv2.Canny(res, 50, 200)
            gauss = cv2.blur(edges, (7, 7), 0) #reduce noise

            contours, hierarchy = cv2.findContours(gauss.copy(), cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)

            if contours and cv2.contourArea(max(contours,key = cv2.contourArea)) > noiseth:
                color=Dict[i]
                flag=1
                color_list=l[i]
                break

        if cv2.waitKey(1) and flag==1:
            cv2.destroyAllWindows()
            draw(color_list,color)
            break                

def draw(color_list,color):
    global canvas
    x1,y1=0,0

    while(1):       
        _, frame = cap.read()

        if canvas is None:
            canvas=np.zeros_like(frame)

        # Converts images from BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_yellow=np.array([21, 130, 200], np.uint8)
        upper_yellow=np.array([40, 255, 255], np.uint8)

        # lower_green = np.array([55, 130, 92], np.uint8)
        # upper_green = np.array([90, 255, 255], np.uint8)

        color_lower=np.array(color_list[0], np.uint8)
        color_upper=np.array(color_list[1], np.uint8)
    
        # creates a mask of green coloured objects found in the frame.
        mask = cv2.inRange(hsv, color_lower, color_upper)
        mask2 = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
        #To clear the canvas
        if np.any(mask2):
            res = cv2.bitwise_and(frame,frame, mask= mask2)

            edges = cv2.Canny(res, 50, 200)
            gauss = cv2.blur(edges, (7, 7), 0) #reduce noise

            contours, hierarchy = cv2.findContours(gauss.copy(), cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)

            if contours and cv2.contourArea(max(contours,key = cv2.contourArea)) > noiseth:
                canvas=np.zeros_like(frame)

        # the green coloured objects are highlighted
        res = cv2.bitwise_and(frame,frame, mask= mask)

        edges = cv2.Canny(res, 50, 200)
        gauss = cv2.blur(edges, (7, 7), 0) #reduce noise

        contours, hierarchy = cv2.findContours(gauss.copy(), cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)
        #cv2.drawContours(frame, contours, -1, (0, 0, 255), thickness=2)
        if contours and cv2.contourArea(max(contours,key = cv2.contourArea)) > noiseth:
            c = max(contours, key = cv2.contourArea)
            x2,y2,w,h = cv2.boundingRect(c)  #boundingReact- highlight the region of contour
            area = cv2.contourArea(c)

            if x1 == 0 and y1 == 0:
                x1,y1= x2,y2
            
            else:
                canvas = cv2.line(canvas, (x1,y1),(x2,y2), color_list[0], thickness=6)
            
            x1,y1= x2,y2

        else:
            x1,y1 =0,0
        final = cv2.add(frame,canvas)
        #stacked = np.hstack((canvas,frame))

        #cv2.imshow('mask',mask)
        cv2.imshow('res',frame)
        font = cv2.FONT_HERSHEY_SIMPLEX
        test=cv2.flip(canvas,1)
        cv2.putText(test, 'Detected: '+color, (1,450), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.resizeWindow('final', 1000,600)
        cv2.imshow('final',test)

        # cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
        # imS = cv2.resize(final, (1790, 970))

    
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
find()      
# cv2.destroyAllWindows()
 
# cap.release()