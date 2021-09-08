import numpy as np
import cv2
import os
import Facerecognition as fr
print(fr)

test_img = cv2.imread(r"C:\Users\cpkha\OneDrive\Desktop\IPMV Project\CHinmay(1).HEIC")

faces_detected, gray_img = fr.faceDetection(test_img)
print("Face Detected :", faces_detected)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r"C:\Users\cpkha\PycharmProjects\IPMV Project\trainingData.yml")

name = {0 : "Chinmay"}

for face in faces_detected:
    (x, y, w, h) = face
    roi_gray = gray_img[y : y+w, x : x+h]
    label, confidence = face_recognizer.predict(roi_gray)
    print("confidence :", confidence)
    print("Label :", label)
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    if (confidence > 47):
        fr.put_text(test_img, 'unknown', x, y)
        continue

    else :
        fr.put_text(test_img, predicted_name, x, y)
resized_img = cv2.resize(test_img, (700, 500))

cv2.imshow("Face Detection :", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()