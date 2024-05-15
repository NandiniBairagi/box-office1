import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import sys

# parameters for loading data and images
detection_model_path = 'cascade/haarcascade_frontalface_alt.xml'
emotion_model_path = 'models/model.hdf5'

# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

camera = cv2.VideoCapture(0)

if camera.isOpened():
    while True:
        _, frame = camera.read()
        orig_frame = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(frame, scaleFactor=1.4, minNeighbors=5, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) != 0:
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
            roi = frame[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            print(label)
            cv2.putText(orig_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
        cv2.imshow('FER', orig_frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            camera.release()
            cv2.destroyAllWindows()
else:
    sys.exit("Camera isn't working.")
    
    
    
    
    
    
    
    
    
