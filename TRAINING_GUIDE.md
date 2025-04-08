# Training Guide: Custom Sign Language Recognition Model

**Version:** 1.0
**Date:** 2023-10-27 (Updated 2025-04-07)

## 1. Introduction

This document provides guidance on how to train your own sign language recognition model, similar to the type that could be used with the accompanying Flask application (`app.py`).

**Note:** The provided `app.py` currently uses the pre-trained `RavenOnur/Sign-Language` image classification model from Hugging Face for simplicity and immediate demonstration. The steps below outline the process if you wanted to train a custom model, potentially a sequence-based one as originally envisioned in the PRD or a different image classification model.

## 2. Overview of the Training Process

Training a sign language recognition model typically involves these stages:

1.  **Data Collection/Selection:** Gathering video data of the signs you want to recognize.
2.  **Data Preprocessing:** Extracting relevant features from the videos.
3.  **Model Definition:** Choosing and defining a suitable machine learning model architecture.
4.  **Model Training:** Training the model on the preprocessed data.
5.  **Model Evaluation:** Assessing the model's performance.
6.  **Model Saving:** Saving the trained model and its label mapping for inference.

## 3. Data Collection / Selection

*   **Choose Your Signs:** Decide on a specific, limited set of signs you want the model to recognize (e.g., "hello", "thanks", "yes", "no", "help").
*   **Find Existing Datasets:** Explore publicly available sign language datasets. A common starting point is:
    *   **WLASL (Word-Level American Sign Language):** Contains a large vocabulary but requires careful selection and preprocessing. ([https://github.com/dxli94/WLASL](https://github.com/dxli94/WLASL))
*   **Create a Custom Dataset:** This is often necessary for specific signs or better control.
    *   **Recording:** Record multiple video examples (15-50+) of each sign.
    *   **Diversity:** Include different signers, backgrounds, lighting conditions, and camera angles if possible.
    *   **Consistency:** Ensure signs are performed clearly and consistently within the videos.
    *   **Organization:** Organize videos into folders named after the sign they represent (e.g., `data/hello/vid1.mp4`, `data/thanks/vid1.mp4`).

## 4. Data Preprocessing

This step transforms raw video data into a format suitable for the model. The method depends heavily on the chosen model type (sequence vs. image classification).

**Option A: Feature Extraction for Sequence Models (e.g., LSTM, GRU, Transformer)**

This was the approach suggested in the original PRD.

1.  **Keypoint Extraction:** Use libraries like MediaPipe (Holistic or Hands) to extract keypoints (landmarks) for hands, pose, and/or face for each frame of every video.
    ```python
    # Example using MediaPipe Holistic
    import mediapipe as mp
    import cv2
    import numpy as np

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Inside a loop processing video frames (frame):
    # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # results = holistic.process(image)
    # Extract keypoints (e.g., pose, left_hand, right_hand)
    # pose_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else []
    # ... extract others ...
    # Flatten and concatenate coordinates into a single feature vector per frame
    # keypoints = np.concatenate([...]).flatten() if landmarks exist else np.zeros(...)
    ```
2.  **Sequence Creation:** Group the keypoint vectors from consecutive frames into sequences of a fixed length (e.g., 30 frames). Pad or truncate sequences as needed.
3.  **Labeling:** Assign a numerical label (0, 1, 2...) to each sequence corresponding to the sign performed.
4.  **Saving:** Save the sequences and their labels (e.g., as NumPy arrays or in TFRecord format).

**Option B: Frame Preparation for Image Classification Models (e.g., CNN, Vision Transformer)**

This approach aligns with the `RavenOnur/Sign-Language` model used in `app.py`.

1.  **Frame Extraction:** Extract individual frames from the sign language videos. You might select keyframes or sample frames at regular intervals.
2.  **Image Preprocessing:** Resize, normalize, and potentially augment the extracted frames according to the requirements of the chosen image classification model architecture. Hugging Face processors often handle this.
3.  **Labeling:** Assign a numerical label to each frame based on the sign it belongs to.
4.  **Dataset Creation:** Structure the data in a format suitable for the training framework (e.g., PyTorch `Dataset`, TensorFlow `tf.data.Dataset`).

## 5. Model Definition

Choose a model architecture suitable for your preprocessed data.

*   **Sequence Models (for Keypoint Sequences):**
    *   **LSTM (Long Short-Term Memory):** Good for capturing temporal dependencies.
    *   **GRU (Gated Recurrent Unit):** Similar to LSTM, often slightly simpler.
    *   **Transformer:** Can capture long-range dependencies effectively.
    *   *Frameworks:* TensorFlow/Keras, PyTorch.
*   **Image Classification Models (for Frames):**
    *   **CNN (Convolutional Neural Network):** Standard for image tasks (e.g., ResNet, MobileNet).
    *   **Vision Transformer (ViT):** Newer architecture showing strong performance.
    *   *Frameworks:* TensorFlow/Keras, PyTorch, Hugging Face `transformers`.

Define the model architecture using your chosen framework, ensuring the input layer matches your preprocessed data shape and the output layer has neurons equal to the number of signs you are classifying, typically with a softmax activation.

## 6. Model Training

1.  **Split Data:** Divide your preprocessed data into training, validation, and test sets.
2.  **Compile/Configure:** Define the loss function (e.g., `CategoricalCrossentropy`), optimizer (e.g., `Adam`), and metrics (e.g., `accuracy`).
3.  **Train:** Use the training framework's `fit` (Keras) or training loop (PyTorch) to train the model on the training data, using the validation set to monitor performance and prevent overfitting (e.g., using callbacks like `EarlyStopping`).
4.  **Hyperparameter Tuning:** Experiment with learning rate, batch size, sequence length (if applicable), model layers, etc., to optimize performance.

## 7. Model Evaluation

*   Evaluate the trained model on the held-out test set to get an unbiased estimate of its performance (accuracy, precision, recall, F1-score).
*   Analyze misclassifications to understand where the model struggles.

## 8. Model Saving

*   **Save the Model:** Save the trained model's architecture and weights.
    *   *Keras:* `model.save('sign_model.h5')` or `model.save('saved_model_dir')`
    *   *PyTorch:* `torch.save(model.state_dict(), 'sign_model.pth')`
    *   *Hugging Face:* `model.save_pretrained('my_sign_model_directory')`, `processor.save_pretrained('my_sign_model_directory')`
*   **Save Label Mapping:** Create and save a mapping from the numerical labels (0, 1, 2...) used during training to the human-readable sign names ("hello", "thanks", ...). A simple JSON file is common:
    ```json
    {
      "0": "hello",
      "1": "thanks",
      "2": "yes",
      "3": "no"
    }
    ```
    Save this as `labels.json` or similar. (Note: Hugging Face models often store this in their `config.json` under `id2label`).

These saved files (`sign_model.h5`/`.pth`/directory and `labels.json`) are what the inference application (`app.py`) would load to perform real-time predictions.
