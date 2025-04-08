import cv2
from flask import Flask, render_template, Response
import torch
from PIL import Image
import numpy as np
import mediapipe as mp
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models

# --- MediaPipe Initialization ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Single-input MobileNetV2 model ---
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 24)
model = model.to(device)

try:
    model.load_state_dict(torch.load('asl_mobilenetv2_augmented.pth', map_location=device))
    model.eval()
    print("Fine-tuned MobileNetV2 model loaded successfully.")
except Exception as e:
    print(f"Error loading ASL model: {e}")
    model = None

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

    if cap is None or not cap.isOpened() or model is None:
        print("Webcam or model not available.")
        return

    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    transform_img = transform
    frame_count = 0
    last_prediction_text = "No Prediction"

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
            landmark_tensor = torch.tensor(landmark_list, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            landmark_tensor = torch.zeros((1, 21*3), dtype=torch.float32).to(device)

        # Throttle inference
        if frame_count % 5 == 0:
            try:
                pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                img_tensor = transform_img(pil_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_idx = torch.argmax(probs, dim=1).item()
                    confidence = probs[0, pred_idx].item()
                # Map index to letter
                alphabet = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
                pred_label = alphabet[pred_idx] if pred_idx < len(alphabet) else "?"
                last_prediction_text = f"{pred_label} ({confidence:.2f})"
            except Exception as e:
                print(f"Error during prediction: {e}")
                last_prediction_text = "Error"

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
