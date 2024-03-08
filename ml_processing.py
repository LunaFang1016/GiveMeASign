import cv2
import numpy as np
import time
import mediapipe as mp
from keras.models import load_model
from try_with_hands_and_pose import *

# Load your ML model
model = load_model('sign_language_recognition_model2.h5')

def mediapipe_detection(frame, model, hands_pose): 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = hands_pose.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame, results 

def preprocess_image(image):
    if image is not None:
        resized_image = cv2.resize(image, (30, 225))
        grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        normalized_image = grayscale_image / 255.0
        input_data = normalized_image.reshape(30, 225)
        return input_data
    else: return np.zeros((30, 225))

def extract_hand_gesture(frame, hand_landmarks):
    if hand_landmarks is not None:
        min_x = int(min(landmark.x for landmark in hand_landmarks.landmark) * frame.shape[1])
        max_x = int(max(landmark.x for landmark in hand_landmarks.landmark) * frame.shape[1])
        min_y = int(min(landmark.y for landmark in hand_landmarks.landmark) * frame.shape[0])
        max_y = int(max(landmark.y for landmark in hand_landmarks.landmark) * frame.shape[0])
        hand_gesture = frame[min_y:max_y, min_x:max_x]
        return hand_gesture
    else: return None

def process_prediction(prediction):
    class_index = np.argmax(prediction)
    labels = ["hello", "howAre", "love", "mask", "no", "please", "sorry", "thanks", "wear", "you"]
    predicted_label = labels[class_index]
    for i, prob in enumerate(prediction[0]):
        print(f"{labels[i]}: {prob:.4f}")
    
    predicted_label = labels[class_index]
    print(f"Predicted label: {predicted_label}")
    return predicted_label


cap = cv2.VideoCapture(0)
start_time = time.time()
frame_counter = 0

with mp.solutions.holistic.Holistic(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands_pose:
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            frame_counter += 1
            fps = frame_counter / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.3f}", (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2, cv2.LINE_AA)

            frame, results = mediapipe_detection(frame, model, hands_pose)
            draw_styled_landmarks(frame, results) 

            if results.left_hand_landmarks or results.right_hand_landmarks:
                for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
                    hand_image = extract_hand_gesture(frame, hand_landmarks)
                    preprocessed_image = preprocess_image(hand_image)
                    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
                    prediction = model.predict(preprocessed_image)
                    predicted_gesture = process_prediction(prediction)
                    cv2.putText(frame, predicted_gesture, (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('MediaPipe Hands and Pose', cv2.flip(frame, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        else:
            print("Ignoring empty camera frame.")