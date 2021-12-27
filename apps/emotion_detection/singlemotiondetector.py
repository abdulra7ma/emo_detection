# Python imports
import warnings
from os.path import abspath, join

# external imports
import cv2
import imutils
import numpy as np
from keras.preprocessing import image
from tensorflow.keras import models
from tensorflow.keras.layers import (
    AveragePooling2D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential, load_model

warnings.filterwarnings("ignore")


class SingleMotionDetector:
    def __init__(self, img, emotion, age, race, gender) -> None:
        self.face_haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.media_dir = "static/processed_imgs"
        self.img = img
        self.emotion = emotion
        self.age = age
        self.race = race
        self.gender = gender

    def __call__(self, *args, **kwds):
        emotion_detecator = self.emotion_detecator(self.img)
        return emotion_detecator

    def emotion_detecator(self, test_img):
        model = load_model("weights/emotion_model.h5")
        emotions = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral",
        ]

        readed_img = cv2.imread(test_img)
        gray_img = cv2.cvtColor(readed_img, cv2.COLOR_BGR2RGB)
        # detect the faces
        face_detected = self.face_haar_cascade.detectMultiScale(gray_img)
        
        for (x, y, w, h) in face_detected:
            cv2.rectangle(
                readed_img, (x, y), (x + w, y + h), (0, 255, 255), thickness=2
            )
            roi_gray = gray_img[
                y : y + w, x : x + h
            ]  # cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray, (48, 48))

            roi = roi_gray.astype('float')/255.0
            # roi = image.img_to_array(roi)
            roi = np.expand_dims(roi, axis=-1)

            prediction = model.predict(roi)[0]

            # find max indexed array
            max_index = np.argmax(prediction)

            predicted_emotion = emotions[max_index]

            cv2.putText(
                readed_img,
                predicted_emotion,
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 88, 255),
                1,
            )

        # resize the width(1000) and the height(700)
        resized_img = cv2.resize(readed_img, (1000, 700))

        new_image = f"{join(self.media_dir, test_img.split('/')[-1].split('.')[0])}.jpg"

        # save the image to the media directory
        cv2.imwrite(new_image, resized_img)

        return new_image

    def load_model(self, model_path):
        """Loads a model saved via model.

        Args:
            model: distenation to the needed model to be used

        """
        print("Loading model start...")
        num_classes = 7

        model = Sequential()

        print("1st convolution layer...")
        # 1st convolution layer
        model.add(Conv2D(64, (5, 5), activation="relu", input_shape=(224, 224, 3)))
        model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
        print("2st convolution layer...")

        # 2nd convolution layer
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
        print("3st convolution layer...")

        # 3rd convolution layer
        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Flatten())

        # fully connected neural networks
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(num_classes, activation="softmax"))
        print("dense layer added")

        model.load_weights("weights/facial_expression_model_weights.h5")
        print("Weights has been successfully loaded")
        return model
