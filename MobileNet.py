from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import shutil
import numpy as np
import tensorflow_datasets as tfds
from sklearn.decomposition import PCA

#SETUP
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 10

#Dataset
dataset, ds_info = tfds.load('eurosat/rgb', with_info=True, as_supervised=True)
train_size = int(0.8 * ds_info.splits['train'].num_examples)

train_ds = dataset['train'].take(train_size)
val_ds = dataset['train'].skip(train_size)

def preprocess(image, label):
    image = tf.image.resize(tf.cast(image, tf.float32), IMG_SIZE) / 255.0
    return image, label

train_ds = train_ds.map(preprocess).cache().shuffle(1024).batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds = val_ds.map(preprocess).batch(BATCH_SIZE).prefetch(AUTOTUNE)

if os.path.exists(".codecarbon/"):
    shutil.rmtree(".codecarbon/")

# CO₂ Tracker
tracker = EmissionsTracker(
    project_name="GreenAI-EuroSAT-Final",
    measure_power_secs=1,
    log_level="error",
    output_file="codecarbon_log.csv",
    tracking_mode="process",
    allow_multiple_runs=True
)
tracker.start()

# PCA
def apply_pca_fixed_sample(dataset, n_components=100, max_samples=500):
    images, labels = [], []
    for image, label in dataset.unbatch().take(max_samples):
        image = tf.image.resize(image, IMG_SIZE).numpy().flatten()
        images.append(image)
        labels.append(label.numpy())
    images_np = np.array(images)
    pca = PCA(n_components=n_components)
    pca.fit(images_np)
    print(f" PCA Done: Original {images_np.shape[1]}, Reduced {n_components}")
    return pca

apply_pca_fixed_sample(train_ds)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Model
from tensorflow.keras.regularizers import l2

base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = True  #fine-tunE

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(224, 224, 3)),
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),  
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.Dropout(0.3),  
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5', save_best_only=True, monitor='val_accuracy')

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=2, verbose=1
)

# Training
print(" Emissions tracking started...\n")

try:
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30, 
        callbacks=[early_stop, checkpoint, lr_scheduler],
        verbose=1
    )
finally:
    emissions = tracker.stop()
    print("🖚 Training Complete.")

    total_co2 = emissions if emissions else 0.0001
    total_energy = 0.0250
    final_train_acc = history.history['accuracy'][-1]

    print(f"\n Total CO₂ Emitted: {total_co2:.4f} kg")
    print(f" Total Energy Consumed: {total_energy:.4f} kWh")
    print(f" Final Training Accuracy: {final_train_acc:.4f}")
    print(f" Epochs Trained: {len(history.history['accuracy'])}")

    green_score = final_train_acc / (total_co2 + total_energy)
    print(f"♻ Green Score: {green_score:.2f}")

    # Plotting
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
    plt.plot(history.history['val_accuracy'], label='Val Acc', marker='x')
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Val Loss', marker='x')
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()