import cv2
import numpy as np

cap = cv2.VideoCapture('video.flv')
ret, frame = cap.read()
cap.set(1, 1000)


#c : position en y    h : hauteur du rectangle
#r : position en x     w : longueur du rectangle
r, h, c, w = 200, 270, 180, 270

track_window = (c, r, w, h)

# ROI = Region Of Interest. It is the region that we're going to color-track
roi = frame[r:r + h, c:c + h]

# convert the Region of Interest from BGR to HSV
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # HSV : Hue Saturation Value

'''
#discard low light value
'''
lowerB = np.array((0., 60., 32.), dtype=np.uint8)
upperB = np.array((180., 255., 255.), dtype=np.uint8)

mask = cv2.inRange(hsv_roi, lowerB, upperB)

# calcHist : calculate the histogram of the array
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Termination criteria : 10 iteration or moved by at least 1 px
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while (1):
    ret, frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply meanshift
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw box
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame, [pts], True, 255, 2)
        cv2.imshow('img2', img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k) + ".jpg", img2)
    else:
        break

cv2.destroyAllWindows()
cap.release()
