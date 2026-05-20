import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATA_PATH = "personalized/MP_DATA_WORDS"
sequence_length = 30
no_sequences = 50

os.makedirs(DATA_PATH, exist_ok=True)

# =========================
# MEDIAPIPE SETUP
# =========================
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

print("\nType word → ENTER")
print("SPACE = start recording")
print("S = skip sequence")
print("ESC = exit\n")

stop_all = False

while True:

    action = input("Enter word: ").upper()

    if action == "EXIT":
        break

    for seq in range(no_sequences):

        if stop_all:
            break

        os.makedirs(f"{DATA_PATH}/{action}/{seq}", exist_ok=True)

        print(f"➡ {action} | Sequence {seq}")

        # =========================
        # WAIT FOR USER TO START
        # =========================
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.putText(frame, f"{action} | Seq:{seq}",
                        (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)

            cv2.putText(frame, "SPACE: Start | S: Skip | ESC: Exit",
                        (10,70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,255), 2)

            cv2.imshow("Collect Words", frame)

            key = cv2.waitKey(1)

            if key == 32:  # SPACE → start recording
                break
            elif key == ord('s'):  # skip
                print("⏭ Skipped")
                continue
            elif key == 27:  # ESC
                stop_all = True
                break

        if stop_all:
            break

        # =========================
        # RECORD SEQUENCE
        # =========================
        for frame_num in range(sequence_length):

            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            result = detector.detect(mp_image)

            features = []

            if result.hand_landmarks:
                hands = sorted(
                    result.hand_landmarks,
                    key=lambda hand: sum([lm.x for lm in hand]) / len(hand)
                )

                for hand in hands:
                    for lm in hand:
                        features.extend([lm.x, lm.y, lm.z])

            while len(features) < 126:
                features.append(0)

            np.save(f"{DATA_PATH}/{action}/{seq}/{frame_num}.npy", features)

            cv2.putText(frame, f"Recording {frame_num+1}/{sequence_length}",
                        (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 2)

            cv2.imshow("Collect Words", frame)

            key = cv2.waitKey(1)

            if key == ord('s'):  # skip mid recording
                print("⏭ Skipped mid-sequence")
                break
            elif key == 27:  # ESC
                stop_all = True
                break

        if stop_all:
            break

    print(f"✅ {action} DONE\n")

    if stop_all:
        print("⛔ Collection stopped")
        break

cap.release()
cv2.destroyAllWindows()