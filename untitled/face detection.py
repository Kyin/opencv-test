import cv2

# load XML classifier
face_cascade = cv2.CascadeClassifier('C:\Media Evaluation\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\Media Evaluation\OpenCV\opencv\sources\data\haarcascades\haarcascade_eye.xml')
print face_cascade
print eye_cascade
# load the image in gray
img = cv2.imread('photo5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# list of rectangle containing the faces detected
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:

    # Trace a blue rectangle around the face
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    # look for the eyes in the face
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0))

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
