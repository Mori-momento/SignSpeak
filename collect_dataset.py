import cv2
import os
import time

# List of ASL letters excluding J and Z (moving signs)
labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y'
]

output_dir = 'dataset'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

for label in labels:
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    print(f"\nGet ready to record sign '{label}'.")
    input("Press Enter when ready...")

    print(f"Recording images for '{label}' in 3 seconds...")
    time.sleep(3)

    count = 0
    max_images = 50  # Number of images per sign
    start_time = time.time()
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
        cv2.imshow("Capture Sign Language Dataset", display_frame)

        # Save frame
        img_path = os.path.join(label_dir, f"{label}_{count+1}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1

        # Wait a short time to avoid duplicates but keep it fast
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    print(f"Captured {count} images for sign '{label}'.")

print("Dataset collection complete.")
cap.release()
cv2.destroyAllWindows()
