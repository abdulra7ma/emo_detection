# Python imports
import os
import warnings

# external imports
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img

warnings.filterwarnings("ignore")

# load model
model = load_model("best_model.h5")

face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class SingleMotionDetector:
    def __init__(self, accumWeight=0.5) -> None:
        self.accumWieght = accumWeight
        # initialize the background model
        self.bg = None

    def update(self, image):
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return
        # update the background model by accumulating the weighted
        # average
        cv2.accumulateWeighted(image, self.bg, self.accumWeight)

    def image_processer(self, test_img):
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi_gray = gray_img[
                y : y + w, x : x + h
            ]  # cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray, (224, 224))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)

            min_pred = predictions[0]

            # find max indexed array
            max_index = np.argmax(min_pred)

            emotions = (
                "angry",
                "disgust",
                "fear",
                "surprise",
                "happy",
                "sad",
                "neutral",
            )
            predicted_emotion = emotions[max_index]

            cv2.putText(
                test_img,
                predicted_emotion,
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow("Facial emotion analysis ", resized_img)
        cv2.imwrite("run/media/")
        return True
