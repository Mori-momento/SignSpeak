import cv2
import mediapipe as mp
import numpy as np
import os
import glob

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
dataset_dir = 'dataset'

for label_dir in os.listdir(dataset_dir):
    label_path = os.path.join(dataset_dir, label_dir)
    if not os.path.isdir(label_path):
        continue

    print(f"Processing label: {label_dir}")
    image_files = glob.glob(os.path.join(label_path, '*.jpg'))

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])
            landmark_array = np.array(landmark_list, dtype=np.float32)
        else:
            # If no hand detected, fill with zeros
            landmark_array = np.zeros(21 * 3, dtype=np.float32)

        # Save landmarks as .npy file
        npy_path = img_path.replace('.jpg', '.npy')
        np.save(npy_path, landmark_array)

print("Landmark extraction complete.")
hands.close()
