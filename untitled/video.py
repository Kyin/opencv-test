import cv2
import numpy as np
#take the 1st frame of the video

#CHANGE THE NAME OF THE FILE AND THE r, h, c, w COORDINATES TO SUIT YOUR NEEDS
cap = cv2.VideoCapture("video.flv")
ret, frame = cap.read()
cap.set(1, 1000)
r, h, c, w = 200,200,200,250

#c : y position     h : height of rectangle
#r : x position     w : width
track_window = (c, r, w, h)

roi = frame[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

lowerB = np.array((0., 60.,32.), dtype=np.uint8)
upperB = np.array((180.,255.,255.), dtype=np.uint8)


#discard low light value
mask = cv2.inRange(hsv_roi, lowerB, upperB)
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

#Setup the termination criteria
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while(1):
    ret, frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        #apply meanshift
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        #draw it
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
        cv2.imshow('img2', img2)


        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)

    else:
        break


cv2.destroyAllWindows()
cap.release()
