# Import all necassary packages
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from mediapipe.tasks.python import vision

model_path = 'hand_landmark.task'
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
     
# TEST: Load the video from specified path
cap = cv2.VideoCapture("video_test.mp4")
# ACTION: Read live video from webcam
# cap = cv2.VideoCapture(0)
start_time = time.time()
frame_counter = 0
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    # TEST: mp4 feed-in case
    while True:
    # ACTION: Live feed-in case
    # while cap.isOpened():
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
            # Flip the frame horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(frame, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        else:
            # TEST: directly exit
            break
            # ACTION: Ignore empty frame
            # print("Ignoring empty camera frame.")
            # continue
        time.sleep(0.1)

# Release all space and windows once done 
cap.release()
# cam.release() 
cv2.destroyAllWindows() 
