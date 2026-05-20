import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

DATA_PATH = "personalized/MP_DATA_WORDS"

# =========================
# LOAD WORDS DYNAMICALLY
# =========================
actions = sorted(os.listdir(DATA_PATH))
print("Detected Words:", actions)

sequences = []
labels = []

label_map = {label: num for num, label in enumerate(actions)}

# =========================
# LOAD DATA
# =========================
for action in actions:
    for seq in os.listdir(f"{DATA_PATH}/{action}"):

        window = []

        for frame_num in range(30):
            path = f"{DATA_PATH}/{action}/{seq}/{frame_num}.npy"

            if os.path.exists(path):
                res = np.load(path)
            else:
                res = np.zeros(126)

            window.append(res)

        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = tf.keras.utils.to_categorical(labels)

print("Dataset Shape:", X.shape)

# =========================
# NORMALIZATION 🔥
# =========================
X = X / np.max(X)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL (IMPROVED)
# =========================
model = Sequential([
    LSTM(128, return_sequences=True, activation='relu', input_shape=(30,126)),
    LSTM(128, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN
# =========================
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# =========================
# EVALUATE
# =========================
loss, accuracy = model.evaluate(X_test, y_test)

print(f"\n✅ Word Model Accuracy: {accuracy*100:.2f}%")

# =========================
# SAVE
# =========================
model.save("personalized/model_words.keras")
np.save("personalized/labels_words.npy", actions)

print("Word model trained ✅")