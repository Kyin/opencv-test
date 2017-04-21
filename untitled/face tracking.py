import cv2

# load XML classifier
face_cascade = cv2.CascadeClassifier(
    'C:\Media Evaluation\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_alt.xml')

# load video
cap = cv2.VideoCapture('video2.mp4')


while 1:
    # get the current frame
    ret, frame = cap.read()

    if ret:

        #detect faces in the frame
        # turn frame gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # look for frontal faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # draw a rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k) + ".jpg", frame)


    else:
        break














