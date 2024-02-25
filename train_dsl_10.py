import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import mediapipe as mp
import cv2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

labels = ["hello", "howAre", "love"]

# Function to load and preprocess the data
def load_data(dataset_folder, mp_holistic):
    X = []
    y = []
    for class_folder in os.listdir(dataset_folder):
        class_path = os.path.join(dataset_folder, class_folder)
        if os.path.isdir(class_path):
            for video_file in os.listdir(class_path):
                video_path = os.path.join(class_path, video_file)
                # Extract features from the video file using MediaPipe
                features = extract_features(video_path, mp_holistic)
                X.append(features)
                y.append(class_folder)
    X = np.array(X)
    y = np.array(y)
    return X, y

# Function to extract features from a video file using MediaPipe
def extract_features(video_path, mp_holistic):
    # Initialize MediaPipe holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Read video frames
        cap = cv2.VideoCapture(video_path)
        features = []
        left_hand_landmarks_empty = np.zeros(21 * 3)  # Assuming 21 landmarks per hand
        right_hand_landmarks_empty = np.zeros(21 * 3)  # Assuming 21 landmarks per hand
        max_feature_length = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Make predictions
            results = holistic.process(frame_rgb)
            # print(results)
            # print(results.left_hand_landmarks)

            # Extract hand landmarks and pose landmarks
            # if results.pose_landmarks and (results.left_hand_landmarks or results.right_hand_landmarks):
            # print("here")
            if results.pose_landmarks:
                pose_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
            # else:
            #     pose_landmarks = {}
            if results.left_hand_landmarks:
                left_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
            else:
                left_hand_landmarks = left_hand_landmarks_empty
            if results.right_hand_landmarks:
                right_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
            else:
                right_hand_landmarks = right_hand_landmarks_empty
            # print("Pose landmarks shape:", pose_landmarks.shape)
            # print("Left hand landmarks shape:", left_hand_landmarks.shape)
            # print("Right hand landmarks shape:", right_hand_landmarks.shape)
            # Concatenate hand landmarks and pose landmarks
            landmarks = np.concatenate([pose_landmarks, left_hand_landmarks, right_hand_landmarks])
            features.append(landmarks)
            max_feature_length = max(max_feature_length, len(landmarks))
        cap.release()

    padded_features = []
    for feature in features:
        if len(feature) < max_feature_length:
            padded_feature = np.pad(feature, (0, max_feature_length - len(feature)), mode='constant')
        else:
            padded_feature = feature
        padded_features.append(padded_feature)

    return np.array(features)

# Load the dataset
dataset_folder = 'datasets/Videos'
mp_holistic = mp.solutions.holistic
X, y = load_data(dataset_folder, mp_holistic)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Convert integer labels to one-hot encoding
num_classes = len(labels)
y_train_onehot = np.eye(num_classes)[y_train_encoded]
y_test_onehot = np.eye(num_classes)[y_test_encoded]

# Create an LSTM model
model = Sequential()
print("X_train shape:", X_train.shape)
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(num_classes, activation='softmax'))  # Adjust the output dimension based on the number of classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_onehot, epochs=10, batch_size=32, validation_data=(X_test, y_test_onehot))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_onehot)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Save the model
model.save('sign_language_recognition_model.h5')


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(np.argmax(y_test_onehot, axis=1), y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()