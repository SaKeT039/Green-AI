import tensorflow as tf 
import tensorflow_datasets as tfds 
import tensorflow_hub as hub 
from tensorflow.keras import layers, models, regularizers 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 
from codecarbon import EmissionsTracker 
import matplotlib.pyplot as plt  

# Start emission tracking 
tracker = EmissionsTracker(project_name="EfficientNetLite0_EuroSAT", measure_power_secs=1) 
tracker.start()

# Load EuroSAT (RGB) dataset 
(ds_train, ds_val), ds_info = tfds.load( 
'eurosat/rgb', 
split=['train[:80%]', 'train[80%:]'], 
shuffle_files=True, 
as_supervised=True, 
with_info=True, 
) 

# Define parameters 
BATCH_SIZE = 32 
IMG_SIZE = (224, 224)  # EfficientNet models expect 224x224 
NUM_CLASSES = ds_info.features['label'].num_classes 
def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE) 
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1] 
    return image, label 

def data_augmentation(image, label): 
    image = tf.image.random_flip_left_right(image) 
    image = tf.image.random_flip_up_down(image) 
    image = tf.image.random_brightness(image, max_delta=0.2) 
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2) 
    return preprocess(image, label) 

# Prepare datasets with data augmentation 
ds_train = ds_train.map(data_augmentation, num_parallel_calls=tf.data.AUTOTUNE) 
ds_train = ds_train.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE) 
ds_val = ds_val.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) 
ds_val = ds_val.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE) 

class EfficientNetLite0Layer(layers.Layer): 
    def __init__(self, hub_url, **kwargs): 
        super(EfficientNetLite0Layer, self).__init__(**kwargs) 
        self.hub_url = hub_url 
        self.base_model = hub.KerasLayer(self.hub_url, input_shape=(224, 224, 3), trainable=False) 
def call(self, inputs): 
    return self.base_model(inputs) 


base_model = EfficientNetLite0Layer("https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2") 

# Build the model using Functional API 
inputs = tf.keras.Input(shape=(224, 224, 3)) 
x = base_model(inputs) 
x = layers.Dropout(0.3)(x)  
x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.02))(x) 
regularization 
x = layers.BatchNormalization()(x)  
x = layers.Dropout(0.3)(x)  
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x) 
model = tf.keras.Model(inputs=inputs, outputs=outputs) 

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  
model.compile( 
optimizer=optimizer, 
loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
metrics=['accuracy'] 
) 

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7) 

history = model.fit( 
ds_train, 
validation_data=ds_val, 
epochs=50, 
callbacks=[early_stopping, reduce_lr]  
) 
emissions = tracker.stop() 

train_accuracy = history.history['accuracy'][-1] 
val_accuracy = history.history['val_accuracy'][-1] 
print("\nTraining Complete!") 
print(f"Training Accuracy: {train_accuracy*100:.2f}%") 
print(f"Validation Accuracy: {val_accuracy*100:.2f}%") 
print(f"CO₂ Emissions: {emissions:.6f} kg") 

green_score = max(0, 100 - emissions * 10) 
print(f"Green Score: {green_score:.2f}/100") 


plt.figure(figsize=(10, 6)) 
plt.plot(history.history['accuracy'], label='Training Accuracy') 
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') 
plt.title('Training and Validation Accuracy') 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.legend() 
plt.grid(True) 
plt.show()