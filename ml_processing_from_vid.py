import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

model = load_model('sign_language_recognition_model3.h5')
labels = ["hello", "howAre", "love", "mask", "no", "please", "sorry", "thanks", "wear", "you"]

def extract_features_from_video(mp_holistic, video_path):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Unable to open video.")
            return
        
        left_hand_landmarks_empty = np.zeros(21 * 3)
        right_hand_landmarks_empty = np.zeros(21 * 3)
        pose_landmarks_empty = np.zeros(33 * 3)
        
        all_landmarks = []  # Store all extracted landmarks
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            if results.pose_landmarks:
                pose_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
            else:
                pose_landmarks = pose_landmarks_empty
            if results.left_hand_landmarks:
                left_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
            else:
                left_hand_landmarks = left_hand_landmarks_empty
            if results.right_hand_landmarks:
                right_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
            else:
                right_hand_landmarks = right_hand_landmarks_empty

            landmarks = np.concatenate([pose_landmarks, left_hand_landmarks, right_hand_landmarks])
            
            all_landmarks.append(landmarks)

        cap.release()
        
        # Convert all_landmarks to numpy array
        all_landmarks = np.array(all_landmarks)

        # Check if there are fewer than 30 frames
        if len(all_landmarks) < 30:
            padding = np.zeros((30 - len(all_landmarks), len(landmarks)))
            all_landmarks = np.concatenate([all_landmarks, padding], axis=0)
        elif len(all_landmarks) > 30:
            # Truncate excess frames
            all_landmarks = all_landmarks[:30]

        # Reshape features to match model input
        features = all_landmarks.reshape(1, 30, -1)
        
        prediction = model.predict(features)
        predicted_class_index = np.argmax(prediction)
        predicted_class = labels[predicted_class_index]

        for i, prob in enumerate(prediction[0]):
                print(f"{labels[i]}: {prob:.4f}")
        
        print("Predicted class:", predicted_class)
  
        cv2.destroyAllWindows()

mp_holistic = mp.solutions.holistic
video_path = 'datasets/extracted_rar/Videos/wear/Movie on 3-13-24 at 6.45 PM #2_cropped.mov'
extract_features_from_video(mp_holistic, video_path)

# for i, prob in enumerate(prediction[0]):
#                 print(f"{labels[i]}: {prob:.4f}")