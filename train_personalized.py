import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
DATA_PATH = "personalized/MP_DATA_PERSONAL"

actions = sorted(os.listdir(DATA_PATH))

data = []
labels = []

label_map = {label: num for num, label in enumerate(actions)}

# =========================
# LOAD DATA
# =========================
for action in actions:
    for file in os.listdir(f"{DATA_PATH}/{action}"):
        data.append(np.load(f"{DATA_PATH}/{action}/{file}"))
        labels.append(label_map[action])

X = np.array(data)
y = np.array(labels)

print("Dataset shape:", X.shape)

# =========================
# TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# MODEL
# =========================
model = Sequential([
    Dense(256, activation='relu', input_shape=(126,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(actions), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN
# =========================
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# =========================
# FINAL EVALUATION
# =========================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\n==============================")
print(f"📊 Final Validation Loss: {loss:.6f}")
print(f"📊 Final Validation Accuracy: {accuracy*100:.2f}%")

# =========================
# LAST EPOCH (OPTIONAL)
# =========================
final_val_loss = history.history['val_loss'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print("\n📈 Last Epoch Metrics:")
print(f"Val Loss: {final_val_loss:.6f}")
print(f"Val Accuracy: {final_val_acc*100:.2f}%")
print("==============================")

# =========================
# SAVE MODEL
# =========================
model.save("personalized/model_personal.keras")
np.save("personalized/labels_personal.npy", actions)

print("Personal model trained ✅")