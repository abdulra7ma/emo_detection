# Python imports
import random
import warnings

# external imports
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# warnings.filterwarnings("ignore")

# load model
model = load_model("models/best_model.h5")


face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


cap = cv2.VideoCapture(0)

while True:
    (
        ret,
        test_img,
    ) = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue

    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
        roi_gray = gray_img[
            y : y + w, x : x + h
        ]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        min_pred = predictions[0]

        # print(min_pred)

        # find max indexed array
        max_index = np.argmax(min_pred)

        emotions = ("angry", "disgust", "fear", "surprise", "happy", "sad", "neutral")
        predicted_emotion = emotions[max_index]

        # print(result["dominant_emotion"])

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

    random_image_num = random.randrange(3, 9999999999)

    # cv2.imwrite(f"run/media/test_img_dir/image_{random_image_num}.png", resized_img)
    cv2.imshow("Facial emotion analysis ", resized_img)

    if cv2.waitKey(10) == ord("q"):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows
