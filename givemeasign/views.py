from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, Http404
from django.views.decorators.csrf import csrf_exempt

# import cv2
import os
import json
import numpy as np
# import mediapipe as mp
from keras.models import load_model
# import time
# from openai import OpenAI
# client = OpenAI()

# model_path =  os.path.join(os.path.dirname(os.path.dirname(__file__)),'givemeasign/assets/signs_10.h5')
# model = load_model(model_path)
# labels = ["hello", "howAre", "love", "mask", "no", "please", "sorry", "thanks", "wear", "you"]

# model = load_model('signs_20.h5')
# labels = ["hello", "howAre", "love", "mask", "no", "please", "sorry", "thanks", "wear", "you",
#           "afternoon", "angry", "bye", "chair", "computer", "confused", "drink", "eat", "evening", 
#           "excited"]

MIN_API_INTERVAL = 5  # Adjust as needed

last_api_call_time = 0
# Create your views here.
def index(request):
    return render(request, 'detect.html')


# def translate_llm(predicted_text):
#     global last_api_call_time

#     # Calculate time elapsed since the last API call
#     # current_time = time.time()
#     # time_elapsed = current_time - last_api_call_time

#     # # If the minimum API interval has not elapsed, wait before making the next call
#     # if time_elapsed < MIN_API_INTERVAL:
#     #     time.sleep(MIN_API_INTERVAL - time_elapsed)

#     prompt = "You will be given a list of words predicted by a sign language recognition model. Translate the sign language sentences into readable English:\n\n"
#     prompt += "For example, if you are given 'hello mask where please', translate it to 'hello please wear your mask'.\n\n"
#     prompt += "If you are not given anything, do not translate anything.\n\n"
#     prompt += "Here is the list of predicted words:\n"
#     prompt += f"{predicted_text}\n"

#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",  # Choose the appropriate GPT model
#         messages=[{"role": "system", "content": prompt}],
#         max_tokens=20,  # Adjust the length of generated text
#         n=10  # Number of texts to translate in a single request
#     )
#     # Update the timestamp of the last API call
#     last_api_call_time = time.time()

#     if response.choices:
#         completion_message = response.choices[0].message
#         if completion_message:
#             text = completion_message.content
#             print(text)
    
#     return text


# def extract_features_from_webcam(mp_holistic):
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             print("Error: Unable to open webcam.")
#             return
        
#         left_hand_landmarks_empty = np.zeros(21 * 3)
#         right_hand_landmarks_empty = np.zeros(21 * 3)
#         pose_landmarks_empty = np.zeros(33 * 3)
        
#         all_landmarks = []
#         frames_processed = 0
#         predicted_words = []
#         last_right_hand_detected_time = time.time()
#         last_left_hand_detected_time = time.time()
#         previous_predicted_text = ''
#         prev_llm = ''
#         curr_llm = ''
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Error: Unable to capture frame.")
#                 break

#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = holistic.process(frame_rgb)

#             if results.pose_landmarks:
#                 pose_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
#                 mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
#             else:
#                 pose_landmarks = pose_landmarks_empty
#             if results.left_hand_landmarks:
#                 last_left_hand_detected_time = time.time()
#                 left_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
#                 mp.solutions.drawing_utils.draw_landmarks(frame, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
#             else:
#                 left_hand_landmarks = left_hand_landmarks_empty
#             if results.right_hand_landmarks:
#                 last_right_hand_detected_time = time.time()
#                 right_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
#                 mp.solutions.drawing_utils.draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
#             else:
#                 right_hand_landmarks = right_hand_landmarks_empty

#             landmarks = np.concatenate([pose_landmarks, left_hand_landmarks, right_hand_landmarks])
            
#             all_landmarks.append(landmarks)
#             frames_processed += 1

#             if len(all_landmarks) >= 30:
#                 all_landmarks_np = np.array(all_landmarks)
#                 features = all_landmarks_np[-30:]
#                 features_reshaped = features.reshape(1, 30, -1)
                
#                 prediction = model.predict(features_reshaped)
#                 predicted_class_index = np.argmax(prediction)
#                 predicted_class = labels[predicted_class_index]

#                 for i, prob in enumerate(prediction[0]):
#                     print(f"{labels[i]}: {prob:.4f}")
                
#                 print("Predicted class:", predicted_class)
#                 predicted_words.append(predicted_class)
#                 all_landmarks.clear()

#             # Check if hands are not detected for 5 seconds
#             if (time.time() - last_right_hand_detected_time > 5) and (time.time() - last_left_hand_detected_time > 5):
#                 print("No hands detected for 5 seconds. Clearing predicted words.")
#                 print(predicted_words)
#                 predicted_words = []
#                 frames_processed = 0
#                 previous_predicted_text = ''
#                 prev_llm = ''
#                 curr_llm = ''
#                 continue

#             # Update previous and current predicted text
#             current_predicted_text = " ".join(predicted_words)
#             if current_predicted_text != previous_predicted_text and current_predicted_text != '':
#                 previous_predicted_text = current_predicted_text
#                 prev_llm = curr_llm
#                 curr_llm = translate_llm(previous_predicted_text)

#             # Draw previous and current translation text
#             cv2.putText(frame, previous_predicted_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             # cv2.putText(frame, prev_llm, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#             print(curr_llm)
#             cv2.putText(frame, curr_llm, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#             cv2.imshow('Sign Language Recognition', frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

@csrf_exempt
def translate(request):
    if request.method == 'POST':
        landmarks = json.loads(request.body)
        landmarks = landmarks.get('landmarks')
        print(landmarks)
        landmarks_np = np.array(landmarks)
        print(landmarks_np.shape)
        # features_reshaped = landmarks_np.reshape(1, 30, -1)
        # print(features_reshaped)
        # Process landmarks using your ML model
        # prediction = model.predict(features_reshaped)
        # predicted_class_index = np.argmax(prediction)
        # predicted_class = labels[predicted_class_index]
        # predicted_words = []
        # for i, prob in enumerate(prediction[0]):
        #     print(f"{labels[i]}: {prob:.4f}")
        predicted_words = "This is prediction test"
        # print("Predicted class:", predicted_class)
        # predicted_words.append(predicted_class)
        # Return prediction as JSON response
        return JsonResponse({'prediction': predicted_words})
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
