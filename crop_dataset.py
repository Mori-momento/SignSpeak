import cv2
import mediapipe as mp
import os
import glob

input_dir = 'dataset'
output_dir = 'dataset_cropped'
os.makedirs(output_dir, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

for label in os.listdir(input_dir):
    label_in_dir = os.path.join(input_dir, label)
    label_out_dir = os.path.join(output_dir, label)
    os.makedirs(label_out_dir, exist_ok=True)
    if not os.path.isdir(label_in_dir):
        continue

    print(f"Processing label: {label}")
    for img_path in glob.glob(os.path.join(label_in_dir, '*.jpg')):
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        h, w, _ = img.shape
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            min_x = max(int(min(x_coords) * w) - 20, 0)
            max_x = min(int(max(x_coords) * w) + 20, w)
            min_y = max(int(min(y_coords) * h) - 20, 0)
            max_y = min(int(max(y_coords) * h) + 20, h)
            cropped = img[min_y:max_y, min_x:max_x]
        else:
            cropped = img  # fallback to full image if no hand detected

        out_path = os.path.join(label_out_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, cropped)

print("Cropping complete. Cropped dataset saved to 'dataset_cropped/'.")
hands.close()
