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

