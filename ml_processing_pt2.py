import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import time

model = load_model('signs_10.h5')
labels = ["hello", "howAre", "love", "mask", "no", "please", "sorry", "thanks", "wear", "you"]

def extract_features_from_webcam(mp_holistic):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Unable to open webcam.")
            return
        
        left_hand_landmarks_empty = np.zeros(21 * 3)
        right_hand_landmarks_empty = np.zeros(21 * 3)
        pose_landmarks_empty = np.zeros(33 * 3)
        
        all_landmarks = []
        frames_processed = 0
        predicted_words = []
        last_right_hand_detected_time = time.time()
        last_left_hand_detected_time = time.time()
        previous_predicted_text = ''  # Store previous predicted text
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            if results.pose_landmarks:
                pose_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
            else:
                pose_landmarks = pose_landmarks_empty
            if results.left_hand_landmarks:
                last_left_hand_detected_time = time.time()
                left_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
                mp.solutions.drawing_utils.draw_landmarks(frame, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            else:
                left_hand_landmarks = left_hand_landmarks_empty
            if results.right_hand_landmarks:
                last_right_hand_detected_time = time.time()
                right_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
                mp.solutions.drawing_utils.draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            else:
                right_hand_landmarks = right_hand_landmarks_empty

            landmarks = np.concatenate([pose_landmarks, left_hand_landmarks, right_hand_landmarks])
            
            all_landmarks.append(landmarks)
            frames_processed += 1

            if len(all_landmarks) >= 30:
                all_landmarks_np = np.array(all_landmarks)
                features = all_landmarks_np[-30:]
                features_reshaped = features.reshape(1, 30, -1)
                
                prediction = model.predict(features_reshaped)
                predicted_class_index = np.argmax(prediction)
                predicted_class = labels[predicted_class_index]

                for i, prob in enumerate(prediction[0]):
                    print(f"{labels[i]}: {prob:.4f}")
                
                print("Predicted class:", predicted_class)
                predicted_words.append(predicted_class)
                all_landmarks.clear()

            # Check if hands are not detected for 5 seconds
            if (time.time() - last_right_hand_detected_time > 5) and (time.time() - last_left_hand_detected_time > 5):
                print("No hands detected for 5 seconds. Clearing predicted words.")
                print(predicted_words)
                predicted_words = []
                frames_processed = 0
                previous_predicted_text = ''  # Clear previous text if hands are not detected
                continue

            # Update previous predicted text only if it has changed
            current_predicted_text = " ".join(predicted_words)
            if current_predicted_text != previous_predicted_text and current_predicted_text != '':
                previous_predicted_text = current_predicted_text

            # Display previous predicted text
            cv2.putText(frame, previous_predicted_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Sign Language Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

mp_holistic = mp.solutions.holistic
extract_features_from_webcam(mp_holistic)
