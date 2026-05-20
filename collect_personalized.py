import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATA_PATH = "personalized/MP_DATA_PERSONAL"
actions = []  # dynamic labels
samples_per_class = 80

os.makedirs(DATA_PATH, exist_ok=True)

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

detector = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

current_action = None
count = 0

print("Press any key to start collecting label (A-Z or custom)")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    result = detector.detect(mp_image)

    key = cv2.waitKey(1) & 0xFF

    if key != 255:
        char = chr(key).upper()

        if char.isalnum():
            current_action = char
            count = 0

            if char not in actions:
                actions.append(char)

            os.makedirs(f"{DATA_PATH}/{char}", exist_ok=True)
            print(f"Collecting {char}...")

        elif key == 27:
            break

    if current_action and result.hand_landmarks:

        hands = sorted(
            result.hand_landmarks,
            key=lambda hand: sum([lm.x for lm in hand]) / len(hand)
        )

        features = []

        for hand in hands:
            for lm in hand:
                features.extend([lm.x, lm.y, lm.z])

        while len(features) < 126:
            features.append(0)

        if len(features) == 126 and count < samples_per_class:
            np.save(f"{DATA_PATH}/{current_action}/{count}.npy", features)
            count += 1

        if count == samples_per_class:
            print(f"{current_action} DONE ✅")
            current_action = None

    cv2.imshow("Collect Personal Data", frame)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()