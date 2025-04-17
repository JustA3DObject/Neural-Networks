import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Dense, AveragePooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Loading and preprocessing
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Enhanced data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)
train_generator = train_datagen.flow(x_train, y_train, batch_size=128)

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow(x_test, y_test, batch_size=128)

# ResNet-20 implementation
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    if stride != 1:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same',
                          kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

inputs = Input(shape=(32, 32, 3))
x = Conv2D(16, 3, padding='same', kernel_initializer='he_normal')(inputs)
x = BatchNormalization()(x)
x = ReLU()(x)

# Stack residual blocks
x = residual_block(x, 16)
x = residual_block(x, 16)
x = residual_block(x, 16)

x = residual_block(x, 32, stride=2)
x = residual_block(x, 32)
x = residual_block(x, 32)

x = residual_block(x, 64, stride=2)
x = residual_block(x, 64)
x = residual_block(x, 64)

x = AveragePooling2D(pool_size=8)(x)
x = Flatten()(x)
x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = Dropout(0.3)(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)

# Learning rate schedule
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 75:
        lr *= 0.1
    elif epoch > 50:
        lr *= 0.5
    return lr

optimizer = optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train with learning rate scheduler
history = model.fit(train_generator, epochs=100, validation_data=test_generator,
                    callbacks=[LearningRateScheduler(lr_schedule)], verbose=1)

# Evaluation
y_pred = np.argmax(model.predict(x_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)
print("Classification Report:\n", classification_report(y_test_labels, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test_labels, y_pred))

# Ploting training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.show()