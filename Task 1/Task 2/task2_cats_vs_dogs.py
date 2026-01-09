# ====================================================
# TASK 2: IMAGE CLASSIFICATION USING DEEP LEARNING
# Dataset: Microsoft Cats vs Dogs (Kaggle)
# Framework: TensorFlow (Keras)
# ====================================================

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# ------------------------------
# DATASET PATH
# ------------------------------
data_dir = "PetImages"

# ------------------------------
# FUNCTION TO CHECK VALID IMAGES
# ------------------------------
def is_valid_image(file_path):
    try:
        img = tf.io.read_file(file_path)
        tf.image.decode_jpeg(img)
        return True
    except:
        return False

# ------------------------------
# COLLECT VALID IMAGE PATHS
# ------------------------------
image_paths = []
labels = []

for label, class_name in enumerate(["Cat", "Dog"]):
    class_dir = os.path.join(data_dir, class_name)
    for file in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file)
        if is_valid_image(file_path):
            image_paths.append(file_path)
            labels.append(label)

print("Valid images found:", len(image_paths))

# ------------------------------
# CREATE DATASET
# ------------------------------
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (128, 128))
    img = img / 255.0
    return img, label

dataset = dataset.map(load_and_preprocess)

# Shuffle & split
dataset = dataset.shuffle(1000)
train_size = int(0.8 * len(image_paths))

train_ds = dataset.take(train_size).batch(32)
val_ds = dataset.skip(train_size).batch(32)

print("Dataset prepared successfully")

# ------------------------------
# BUILD CNN MODEL
# ------------------------------
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# ------------------------------
# COMPILE MODEL
# ------------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ------------------------------
# TRAIN MODEL
# ------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# ------------------------------
# VISUALIZE RESULTS
# ------------------------------
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()
