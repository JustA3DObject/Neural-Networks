import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape and convert to RGB (VGG expects 3 channels)
def reshape_for_vgg(data):
    return np.repeat(data[..., np.newaxis], 3, axis=-1).astype("float32") / 255.0

x_train = reshape_for_vgg(x_train)
x_test = reshape_for_vgg(x_test)

# Pad images to 32x32 (original VGG input size)
x_train = tf.pad(x_train, [[0,0], [2,2], [2,2], [0,0]])
x_test = tf.pad(x_test, [[0,0], [2,2], [2,2], [0,0]])

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False  # MNIST digits shouldn't be flipped horizontally
)
train_generator = train_datagen.flow(x_train, y_train, batch_size=128)

# Build VGG-16 model
def vgg16(input_shape=(32,32,3)):
    model = models.Sequential()
    
    # Block 1
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))
    
    # Block 2
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))
    
    # Block 3
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))
    
    # Block 4
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))
    
    # Block 5
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))
    
    # Classification layers
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

# Learning rate schedule
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 15:
        lr *= 0.1
    elif epoch > 10:
        lr *= 0.5
    return lr

# Create and compile model
model = vgg16()
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    epochs=20,
                    validation_data=(x_test, y_test),
                    callbacks=[LearningRateScheduler(lr_schedule)])

# Evaluate performance
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()