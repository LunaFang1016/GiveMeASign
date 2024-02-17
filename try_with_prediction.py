import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from mediapipe.tasks.python import vision
from keras.models import load_model

model = load_model('smnist.h5')

def preprocess_image(image):
    resized_image = cv2.resize(image, (28, 28))
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    normalized_image = grayscale_image / 255.0
    input_data = normalized_image.reshape(28, 28, 1)
    return input_data

def extract_hand_gesture(frame, hand_landmarks):
    min_x = int(min(landmark.x for landmark in hand_landmarks.landmark) * frame.shape[1])
    max_x = int(max(landmark.x for landmark in hand_landmarks.landmark) * frame.shape[1])
    min_y = int(min(landmark.y for landmark in hand_landmarks.landmark) * frame.shape[0])
    max_y = int(max(landmark.y for landmark in hand_landmarks.landmark) * frame.shape[0])

    hand_gesture = frame[min_y:max_y, min_x:max_x]

    return hand_gesture

def process_prediction(prediction):
    class_index = np.argmax(prediction)
    
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    predicted_label = labels[class_index]
    
    print(predicted_label)
    return predicted_label

mp_model_path = 'hand_landmark.task'
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
     
# TEST: Load the video from specified path
# cap = cv2.VideoCapture("video_test.mp4")
# ACTION: Read live video from webcam
cap = cv2.VideoCapture(0)
start_time = time.time()
frame_counter = 0
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    # TEST: mp4 feed-in case
    # while True:
    # ACTION: Live feed-in case
    while cap.isOpened():
        # reading from frame
        success, frame = cap.read()
        if success:
            frame_counter += 1
            fps = frame_counter / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.3f}", (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2, cv2.LINE_AA)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame)

            # Draw the hand annotations on the frame.
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_image = extract_hand_gesture(frame, hand_landmarks)
                    preprocessed_image = preprocess_image(hand_image)
                    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
                    prediction = model.predict(preprocessed_image)
                    predicted_gesture = process_prediction(prediction)
                    cv2.putText(frame, predicted_gesture, (250,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        # # Display the frame with hand landmarks and predicted gesture label
        # cv2.imshow('MediaPipe Hands', cv2.flip(frame, 1))
        # if cv2.waitKey(5) & 0xFF == 27:
        #     break


            # Flip the frame horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(frame, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        else:
            # TEST: directly exit
            # break
            # ACTION: Ignore empty frame
            print("Ignoring empty camera frame.")
            # continue
        # time.sleep(0.1)

# Release all space and windows once done 
cap.release()
# cam.release() 
cv2.destroyAllWindows() 