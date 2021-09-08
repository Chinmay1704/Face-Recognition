import numpy as np
import cv2
import os
import csv
import Facerecognition

print(Facerecognition)



face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r"C:\Users\cpkha\PycharmProjects\IPMV Project\trainingData.yml")

cap = cv2.VideoCapture(0)
size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
       )
# Define the code and create videowriter object. The output is stored in 'output.avi' file.
#if you want to recognize face from a video then replace 0 with video path

name = {0 : "Chinmay", 1 : "Chaitanya"}
while True:
    ret, test_img = cap.read()
    faces_detected, gray_img = Facerecognition.faceDetection(test_img)
    #print("face Detected :", faces_detected)
    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x,y), (x+w, y+h), (0,255,0), thickness = 5)

    for face in faces_detected:
        (x, y, w, h) = face
        roi_gray = gray_img[y : y+h, x : x+h]
        label, confidence = face_recognizer.predict(roi_gray)
        print("confidence :", confidence)
        print("Label :", label)
        Facerecognition.draw_rect(test_img, face)
        predicted_name = name[label]
        if(confidence > 67):
            Facerecognition.put_text(test_img, 'unknown', x, y)
            continue
        Facerecognition.put_text(test_img, predicted_name, x, y)
        if label == 0:
            print(predicted_name)

    resized_img = cv2.resize(test_img, (700, 500))

    cv2.imshow("Face detected :", resized_img)

    if cv2.waitKey(10) == ord("q"):
        break