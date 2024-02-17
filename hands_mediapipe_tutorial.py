import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# # For webcam input:
# cap = cv2.VideoCapture(0)
# with mp_hands.Hands(
#     model_complexity=0,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as hands:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue

#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = hands.process(image)

#     # Draw the hand annotations on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.multi_hand_landmarks:
#       for hand_landmarks in results.multi_hand_landmarks:
#         mp_drawing.draw_landmarks(
#             image,
#             hand_landmarks,
#             mp_hands.HAND_CONNECTIONS,
#             mp_drawing_styles.get_default_hand_landmarks_style(),
#             mp_drawing_styles.get_default_hand_connections_style())
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()

''' 
Install dependencies 
pip install opencv-python 
pip install mediapipe 
'''
# Import packages 
import cv2 
import mediapipe as mp 

#Build Keypoints using MP Holistic 
mp_holistic = mp.solutions.holistic # Holistic model 
mp_drawing = mp.solutions.drawing_utils # Drawing utilities 

def mediapipe_detection(image, model): 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB 
    image.flags.writeable = False                 # Image is no longer writable 
    results = model.process(image)                 # Make prediction 
    image.flags.writeable = True                 # Image is now writable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR 
    return image, results 
    
def draw_landmarks(image, results): 
    mp_drawing.draw_landmarks( 
    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections 
    mp_drawing.draw_landmarks( 
    image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections 
    mp_drawing.draw_landmarks( 
    image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections 
    
def draw_styled_landmarks(image, results): 
    # Draw pose connections 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2) 
                            ) 
    # Draw left hand connections 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2) 
                            ) 
    # Draw right hand connections 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                            ) 
#Main function 
cap = cv2.VideoCapture(1) 
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic: 
    while cap.isOpened(): 

        # Read feed 
        ret, frame = cap.read() 

        # Make detections 
        image, results = mediapipe_detection(frame, holistic) 
        print(results) 
        
        # Draw landmarks 
        draw_styled_landmarks(image, results) 

        # Show to screen 
        cv2.imshow('OpenCV Feed', image) 

        # Break gracefully 
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break
    cap.release() 
    cv2.destroyAllWindows()