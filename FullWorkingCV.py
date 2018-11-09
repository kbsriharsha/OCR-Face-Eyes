import numpy as np
import cv2
from collections import deque
from keras.models import load_model

cnn_model = load_model('cnn_model.h5')
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

X = 0
Y = 0
W = 350
H = 350
letters = { 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}

while 1:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    _, thresh_out = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    roi = thresh_out[X:X+W, Y:Y+W]
    contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    # if counters are present
    if len(contours) > 0:
        contour = max(contours, key = cv2.contourArea)

        if cv2.contourArea(contour) > 2800:
            x, y, w, h = cv2.boundingRect(contour)
            newImage = roi[y:y + h, x:x + w]
            #cv2.imshow("frame", newImage)
            newImage = cv2.resize(newImage, (28, 28))
            #newImage = np.array(newImage)
            prediction1 = cnn_model.predict(newImage.reshape(1,28,28,1))[0]
            print(prediction1)
            prediction1 = np.argmax(prediction1)
            print(prediction1)
            #print(prediction1)
            

    cv2.rectangle(frame, (X, Y), (X + W, Y + H), (0, 255, 0), 2)
    cv2.putText(frame, "CNN Prediction(430k): " + str(letters[int(prediction1)+1]), (10, 600),
                    cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
    frame1 = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)
    cv2.imshow("OCR-Face-Eyes", frame1)

    #cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
