import os
import tensorflow as tf  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # type: ignore

# 📁 Paths
DATASET_DIR = 'dataset'
MODEL_PATH = 'model/crop_model.h5'
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# 🔁 Data Generators (augmentation + normalization)
image_gen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)

train_gen = image_gen.flow_from_directory(
    DATASET_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = image_gen.flow_from_directory(
    DATASET_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 🧠 CNN Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

# ⚙️ Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🏋️ Train
history = model.fit(train_gen, validation_data=val_gen, epochs=10)

# 💾 Save
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
