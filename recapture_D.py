import cv2
import mediapipe as mp
import numpy as np
import os
import time
import glob

label = 'D'
output_dir = os.path.join('dataset', label)

# Remove existing images and landmarks for 'D'
files = glob.glob(os.path.join(output_dir, '*.jpg')) + glob.glob(os.path.join(output_dir, '*.npy'))
for f in files:
    os.remove(f)
print(f"Cleared existing data for '{label}'.")

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print(f"\nGet ready to record sign '{label}'.")
input("Press Enter when ready...")

print(f"Recording images for '{label}' in 3 seconds...")
time.sleep(3)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

count = 0
max_images = 50
while count < max_images:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        continue

    # Show live feed with label and count
    display_frame = frame.copy()
    cv2.putText(display_frame, f"Sign: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Image: {count+1}/{max_images}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Re-Capture Sign D", display_frame)

    # Save image
    img_path = os.path.join(output_dir, f"{label}_{count+1}.jpg")
    cv2.imwrite(img_path, frame)

    # Extract landmarks
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmark_list = []
        for lm in hand_landmarks.landmark:
            landmark_list.extend([lm.x, lm.y, lm.z])
        landmark_array = np.array(landmark_list, dtype=np.float32)
    else:
        landmark_array = np.zeros(21 * 3, dtype=np.float32)

    npy_path = img_path.replace('.jpg', '.npy')
    np.save(npy_path, landmark_array)

    count += 1

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

print(f"Captured {count} images and landmarks for sign '{label}'.")
cap.release()
cv2.destroyAllWindows()
hands.close()
