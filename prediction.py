import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pygame

eyes_closure_model = load_model("eyes_closure_detection_model.h5")
yawning_detection_model = load_model("yawning_detection_model.h5")

cap = cv2.VideoCapture(0)
pygame.mixer.init()
alarm_sound_eyes_closure = pygame.mixer.Sound("alarm.wav")
alarm_sound_yawning = pygame.mixer.Sound("alarm.wav")
alarm_sound_drowsy = pygame.mixer.Sound("alarm.wav")

def play_alarm_eyes_closure():
    pygame.mixer.Sound.play(alarm_sound_eyes_closure)

def play_alarm_yawning():
    pygame.mixer.Sound.play(alarm_sound_yawning)

def play_alarm_drowsy():
    pygame.mixer.Sound.play(alarm_sound_drowsy)

def stop_alarm():
    pygame.mixer.Sound.stop(alarm_sound_eyes_closure)
    pygame.mixer.Sound.stop(alarm_sound_yawning)
    pygame.mixer.Sound.stop(alarm_sound_drowsy)

def preprocess_image(img):
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, (1, 100, 100, 1))
    return img

# Load Haar cascade classifiers
eyes_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_eye.xml")
left_eye_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_righteye_2splits.xml")
mouth_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_mouth.xml")

count = 0
thickness = 1
alert_active = False
drowsiness_time = 0

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eyes_cascade.detectMultiScale(gray)
    left_eye = left_eye_cascade.detectMultiScale(gray)
    right_eye = right_eye_cascade.detectMultiScale(gray)

    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150, 150), 1)

    # Initialize the variables
    right_eye_closure = [float("inf")]
    left_eye_closure = [float("inf")]

    for (x, y, w, h) in right_eye:
        right_eye_roi = frame[y:y + h, x:x + w]
        right_eye_roi = preprocess_image(right_eye_roi)
        right_eye_closure = eyes_closure_model.predict(right_eye_roi)
        count += 1

    for (x, y, w, h) in left_eye:
        left_eye_roi = frame[y:y + h, x:x + w]
        left_eye_roi = preprocess_image(left_eye_roi)
        left_eye_closure = eyes_closure_model.predict(left_eye_roi)
        count += 1

    if right_eye_closure[0] <= 0.5 and left_eye_closure[0] <= 0.5:
        cv2.putText(frame, "Eyes Closed", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        if not alert_active:
            play_alarm_eyes_closure()
            alert_active = True
        drowsiness_time += 1
    else:
        cv2.putText(frame, "Eyes Open", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        if alert_active:
            stop_alarm()
            alert_active = False
        drowsiness_time = 0

    # Yawning detection
    mouth = mouth_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in mouth:
        mouth_roi = frame[y:y + h, x:x + w]
        mouth_roi = preprocess_image(mouth_roi)
        yawning_closure = yawning_detection_model.predict(mouth_roi)
        if yawning_closure[0] <= 0.5:
            cv2.putText(frame, "Yawning", (int(width / 2) - 150, int(height / 2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2, cv2.LINE_AA)
            if not alert_active:
                play_alarm_yawning()
                alert_active = True

    # Trigger alert for drowsiness
    if drowsiness_time > 10:
        if not alert_active:
            play_alarm_drowsy()
            alert_active = True
            cv2.putText(frame, "Alert: You're Drowsy!", (int(width / 2) - 200, int(height / 2) + 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Driver drowsiness and yawning detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
