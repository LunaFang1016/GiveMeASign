# Import all necassary packages
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from mediapipe.tasks.python import vision

# model_path = 'hand_landmark.task'
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands_pose = mp.solutions.holistic

def mediapipe_detection(frame, model): 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB 
    frame.flags.writeable = False                 # frame is no longer writable 
    results = model.process(frame)                 # Make prediction 
    frame.flags.writeable = True                 # frame is now writable 
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR 
    return frame, results 

def draw_landmarks(image, results): 
    mp_drawing.draw_landmarks( 
    image, results.pose_landmarks, mp_hands_pose.POSE_CONNECTIONS) # Draw pose connections 
    mp_drawing.draw_landmarks( 
    image, results.left_hand_landmarks, mp_hands_pose.HAND_CONNECTIONS) # Draw left hand connections 
    mp_drawing.draw_landmarks( 
    image, results.right_hand_landmarks, mp_hands_pose.HAND_CONNECTIONS) # Draw right hand connections 

def draw_styled_landmarks(image, results): 
    # Draw pose connections 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_hands_pose.POSE_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2) 
                            ) 
    # Draw left hand connections 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_hands_pose.HAND_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2) 
                            ) 
    # Draw right hand connections 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_hands_pose.HAND_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                            ) 

# # TEST: Load the video from specified path
# # cap = cv2.VideoCapture("video_test.mp4")
# # ACTION: Read live video from webcam
# cap = cv2.VideoCapture(1)
# start_time = time.time()
# frame_counter = 0
# with mp_hands_pose.Holistic(
#     model_complexity=0,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as hands_pose:
#     # TEST: mp4 feed-in case
#     # while True:
#     # ACTION: Live feed-in case
#     while cap.isOpened():
#         # reading from frame
#         success, frame = cap.read()
#         if success:
#             frame_counter += 1
#             fps = frame_counter / (time.time() - start_time)
#             cv2.putText(frame, f"FPS: {fps:.3f}", (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2, cv2.LINE_AA)

#             # Make detections 
#             frame, results = mediapipe_detection(frame, hands_pose) 
#             # print(results) 
        
#             # Draw landmarks 
#             draw_styled_landmarks(frame, results) 

#             # Flip the frame horizontally for a selfie-view display.
#             cv2.imshow('MediaPipe Hands and Pose', cv2.flip(frame, 1))
#             if cv2.waitKey(5) & 0xFF == 27:
#                 break
#         else:
#             # TEST: directly exit
#             # break
#             # ACTION: Ignore empty frame
#             print("Ignoring empty camera frame.")
#             # continue
#         # time.sleep(0.1)

# # Release all space and windows once done 
# cap.release()
# # cam.release() 
# cv2.destroyAllWindows() 