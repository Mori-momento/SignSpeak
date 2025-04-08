import cv2
from flask import Flask, render_template, Response
import torch
from PIL import Image
import numpy as np
import mediapipe as mp
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
import joblib
from tensorflow.keras.models import load_model

def normalize_landmarks(landmarks):
    # Center to wrist (landmark 0)
    wrist = landmarks[0]
    landmarks -= wrist
    # Scale normalization
    max_value = np.max(np.abs(landmarks))
    if max_value > 0:
        landmarks /= max_value
    return landmarks.flatten()

# --- MediaPipe Initialization ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load SVM model ---
try:
    svm_model = joblib.load('asl_svm_model.pkl')
    print("SVM model loaded successfully.")
except Exception as e:
    print(f"Error loading SVM model: {e}")
    svm_model = None

# --- Load CNN model ---
try:
    cnn_model = load_model('asl_cnn_model.h5')
    print("CNN model loaded successfully.")
except Exception as e:
    print(f"Error loading CNN model: {e}")
    cnn_model = None

# --- Image transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Webcam Initialization ---
try:
    cap = cv2.VideoCapture(0) # Use 0 for the default webcam
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    print("Webcam initialized successfully.")
except Exception as e:
    print(f"Error initializing webcam: {e}")
    cap = None # Indicate that webcam failed

def generate_frames():
    """Generator function with multi-modal model inference."""
    global cap, model

    if cap is None or not cap.isOpened():
        print("Webcam not available.")
        return

    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    transform_img = transform
    frame_count = 0
    last_prediction_text = "No Prediction"
    transcript = ""
    no_hand_counter = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from webcam.")
            break

        frame_count += 1

        # Extract landmarks
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        # Draw right hand landmarks
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=results.right_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

        # Crop to right hand
        cropped_image = frame
        if results.right_hand_landmarks:
            x_coords = [lm.x for lm in results.right_hand_landmarks.landmark]
            y_coords = [lm.y for lm in results.right_hand_landmarks.landmark]
            h, w, _ = frame.shape
            min_x = max(int(min(x_coords) * w) - 20, 0)
            max_x = min(int(max(x_coords) * w) + 20, w)
            min_y = max(int(min(y_coords) * h) - 20, 0)
            max_y = min(int(max(y_coords) * h) + 20, h)
            cropped_image = frame[min_y:max_y, min_x:max_x]


        # Prepare landmarks tensor
        if results.right_hand_landmarks:
            landmark_list = []
            for lm in results.right_hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])
            landmark_array = np.array(landmark_list, dtype=np.float32).reshape(21,3)
            norm_landmarks = normalize_landmarks(landmark_array)
            landmark_np = norm_landmarks.reshape(1, -1)
            landmark_tensor = torch.tensor(norm_landmarks, dtype=torch.float32).unsqueeze(0).to(device)

            # Throttle inference
            if frame_count % 5 == 0:
                try:
                    # Log normalized landmark input
                    print("Normalized landmark input:", norm_landmarks.tolist())

                    # SVM prediction
                    svm_probs = svm_model.predict_proba(landmark_np)
                    svm_pred_idx = np.argmax(svm_probs, axis=1)[0]
                    svm_conf = np.max(svm_probs)
                    svm_alphabet = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
                    svm_label = svm_alphabet[svm_pred_idx] if svm_pred_idx < len(svm_alphabet) else "?"

                    # CNN prediction
                    landmark_cnn_input = landmark_np  # shape (1,63)
                    cnn_probs = cnn_model.predict(landmark_cnn_input, verbose=0)
                    cnn_pred_idx = np.argmax(cnn_probs, axis=1)[0]
                    cnn_conf = np.max(cnn_probs)
                    cnn_label = svm_alphabet[cnn_pred_idx] if cnn_pred_idx < len(svm_alphabet) else "?"

                    # Ensemble: pick model with higher confidence
                    if svm_label == cnn_label:
                        final_letter = svm_label
                    elif svm_conf >= cnn_conf:
                        final_letter = svm_label
                    else:
                        final_letter = cnn_label

                    last_prediction_text = f"SVM: {svm_label} ({svm_conf:.2f}) | CNN: {cnn_label} ({cnn_conf:.2f})"
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    last_prediction_text = "Error"
        else:
            landmark_tensor = torch.zeros((1, 21*3), dtype=torch.float32).to(device)
            no_hand_counter += 1
            if no_hand_counter > 30:
                transcript = ""
            last_prediction_text = "No hand detected"

        # If hand detected, update transcript
        if "No hand detected" not in last_prediction_text:
            no_hand_counter = 0
            # Extract ensemble letter
            parts = last_prediction_text.split("|")
            if len(parts) == 2:
                svm_letter = parts[0].split(":")[1].strip()
                cnn_letter = parts[1].split(":")[1].strip()
                if svm_letter == cnn_letter:
                    letter = svm_letter
                else:
                    letter = svm_letter  # or cnn_letter or majority vote
                # Append if new letter
                if len(transcript) == 0 or transcript[-1] != letter:
                    transcript += letter

        # Draw prediction text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 255, 0)
        thickness = 2
        org = (50, frame.shape[0] - 30)
        (text_width, text_height), baseline = cv2.getTextSize(last_prediction_text, font, font_scale, thickness)
        cv2.rectangle(frame, (org[0] - 10, org[1] + baseline - text_height - 10),
                      (org[0] + text_width + 10, org[1] + baseline + 10),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, last_prediction_text, org, font, font_scale, color, thickness, cv2.LINE_AA)

        # Draw transcript on right
        x_offset = frame.shape[1] - 300
        y_offset = 50
        cv2.rectangle(frame, (x_offset - 10, y_offset - 40), (frame.shape[1] - 10, y_offset + 40), (0,0,0), -1)
        cv2.putText(frame, "Transcript:", (x_offset, y_offset), font, 0.8, (255,255,255), 2)
        cv2.putText(frame, transcript, (x_offset, y_offset + 30), font, 0.8, (255,255,255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode frame.")
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # IMPORTANT: Set debug=False for any kind of deployment or sharing
    # Use host='0.0.0.0' to make it accessible on your network
    app.run(debug=True, host='0.0.0.0', port=5000)
