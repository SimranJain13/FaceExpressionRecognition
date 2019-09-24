from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import cv2
import numpy as np
import sys
import glob
#import argparse

def MainFunction(filename):
    # parameters for loading data and images
    detection_model_path = "C:\\Users\\Hp\\PycharmProjects\\FaceEmotion_ID\\haarcascade_files\\haarcascade_frontalface_default.xml"
    emotion_model_path = 'C:\\Users\Hp\\PycharmProjects\\FaceEmotion_ID\\models_mini_XCEPTION.12-0.59.hdf5'
    # img_path = sys.argv[1]
    cv_img = []
    img_path = filename
    # hyper-parameters for bounding boxes shape
    # loading models
    face_detection = cv2.CascadeClassifier(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

    # reading the frame
    orig_frame = cv2.imread(img_path)
    frame = cv2.imread(img_path, 0)
    faces = face_detection.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        roi = frame[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = preds
        m = []
        for i in range(0,6):
            print("{} : {}".format(EMOTIONS[i],preds[i]*100))
            m.append({EMOTIONS[i]:abs(preds[i]*100)})
            #cv2.putText(orig_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            #cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

    return m
    #cv2.imwrite('C:\\Users\\Hp\\PycharmProjects\\FaceEmotion_ID\\test_output\\' + img_path.split('\\')[-1],
                #orig_frame)
    #if (cv2.waitKey(5000) & 0xFF == ord('q')):
        #sys.exit("Thanks")
    #cv2.destroyAllWindows()









