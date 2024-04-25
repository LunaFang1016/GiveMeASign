from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.middleware.csrf import get_token

# import cv2
import os
import json
import numpy as np
import time
# import mediapipe as mp
from keras.models import load_model
from openai import OpenAI
# client = OpenAI()

# model_path =  os.path.join(os.path.dirname(os.path.dirname(__file__)),'givemeasign/assets/signs_10.h5')
model_path = "givemeasign/assets/signs_10.h5"
model = load_model(model_path)
labels = ["hello", "howAre", "love", "mask", "no", "please", "sorry", "thanks", "wear", "you"]

predicted_words = []

MIN_API_INTERVAL = 5  # Adjust as needed

client = OpenAI()

last_api_call_time = 0
# Create your views here.
def index(request):
    return render(request, 'detect.html')


def translate_llm(predicted_text):
    global last_api_call_time

    # Calculate time elapsed since the last API call
    current_time = time.time()
    time_elapsed = current_time - last_api_call_time

    # If the minimum API interval has not elapsed, wait before making the next call
    if time_elapsed < MIN_API_INTERVAL:
        time.sleep(MIN_API_INTERVAL - time_elapsed)

    prompt = "You will be given a list of words predicted by a sign language recognition model. Translate the sign language sentences into readable English:\n\n"
    prompt += "For example, if you are given 'hello mask where please', translate it to 'hello please wear your mask'.\n\n"
    prompt += "If you are not given anything, do not translate anything.\n\n"
    prompt += "Here is the list of predicted words:\n"
    prompt += f"{predicted_text}\n"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Choose the appropriate GPT model
        messages=[{"role": "system", "content": prompt}],
        max_tokens=20,  # Adjust the length of generated text
        n=10  # Number of texts to translate in a single request
    )
    # Update the timestamp of the last API call
    last_api_call_time = time.time()

    if response.choices:
        completion_message = response.choices[0].message
        if completion_message:
            text = completion_message.content
            print(text)
    
    return text
                

previous_predicted_text = ''
hands_empty_interval = 0

def get_llm():
    global previous_predicted_text
    global hands_empty_interval
    global predicted_words
    curr_llm = ''
    if hands_empty_interval > 150:
        print("end of sentence")
        # print("No hands detected for 5 seconds. Clearing predicted words.")
        # print(predicted_words)
        predicted_words = []
        frames_processed = 0
        hands_empty_interval = 0
        previous_predicted_text = ''
        prev_llm = ''
        curr_llm = ''
        return curr_llm
    else:
        # Update previous and current predicted text
        current_predicted_text = " ".join(predicted_words)
        if current_predicted_text != previous_predicted_text and current_predicted_text != '':
            previous_predicted_text = current_predicted_text
            prev_llm = curr_llm
            curr_llm = translate_llm(previous_predicted_text)
    return curr_llm

def predict_words(landmarks_data):
    num_frames = len(landmarks_data)

    all_landmarks = []

    for frame_index in range(num_frames):
        # print(frame_index)
        frame_landmarks = landmarks_data[frame_index]

        # Extract pose landmarks (landmarks 43-75)
        pose_landmarks = np.array([lm for lm in frame_landmarks[:33]]).flatten()
        # print("pose" , pose_landmarks)

        # Extract hand landmarks (landmarks 1-42)
        hand_landmarks = np.array([lm for lm in frame_landmarks[33:75]]).flatten()
        if np.any(hand_landmarks):
            global hands_empty_interval
            hands_empty_interval += 1
        
        landmarks_per_frame = np.concatenate([pose_landmarks, hand_landmarks])
        
        all_landmarks.append(landmarks_per_frame)
        # print("len", len(all_landmarks))

    if len(all_landmarks) >= 30:
        # print("am i here")
        all_landmarks_np = np.array(all_landmarks)
        features = all_landmarks_np[-30:]
        features_reshaped = features.reshape(1, 30, -1)
        
        prediction = model.predict(features_reshaped)
        predicted_class_index = np.argmax(prediction)
        predicted_class = labels[predicted_class_index]

        for i, prob in enumerate(prediction[0]):
            print(f"{labels[i]}: {prob:.4f}")
        
        print("Predicted class:", predicted_class)
        global predicted_words
        print("predicted_words here", predicted_words)
        if predicted_words == [] or (len(predicted_words) >= 1 and (predicted_class != predicted_words[-1])):
            predicted_words.append(predicted_class)
        all_landmarks.clear()
        return predicted_class


# @csrf_exempt
def translate(request):
    if request.method == 'POST':
        landmarks = json.loads(request.body)
        landmarks = landmarks.get('landmarks')
        # end_of_sentence = landmarks.get('eol')
        # print(landmarks)
        # landmarks_np = np.array(landmarks)
        # print(landmarks_np.shape)
        
        predicted_class = predict_words(landmarks)
        sentence = get_llm()

        
        
        return JsonResponse({'prediction': predicted_class, 'predicted_sentence': sentence})
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

def home(request):
    return render(request, 'home.html')

def instructions(request):
    return render(request, 'instructions.html')