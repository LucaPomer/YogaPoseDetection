# Setup train and test splits
import keras
from keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

from scripts.machineLearning.ml_data_for_classification import MlDataForModelTraining, MlDataForModelTesting

train_data_angles = MlDataForModelTraining(
    '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/experiments/own_split_test/csv_data_files/train_data_angles_with_flipped.csv')
test_data_angles = MlDataForModelTesting('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/experiments/own_split_test/csv_data_files/test_data_angles.csv')

x_train = train_data_angles.train_data
y_train = train_data_angles.train_labels

x_test = test_data_angles.test_data
y_test = test_data_angles.test_labels

# Inspect train data
print("Training data shape: ", x_train.shape)  # (60000, 28, 28) -- 60000 images, each 28x28 pixels



# Inspect test data
print("Test data shape: ", x_test.shape)  # (10000, 28, 28) -- 10000 images, each 28x28 pixels



# Inspect labels
print("Training labels shape: ", y_train.shape)  # (60000, 28, 28) -- 60000 numbers from the range 0, 1, ... 9
print("Test labels shape: ", y_test.shape)  # (10000, 28, 28) -- 10000 numbers from the range 0, 1, ... 9
print("First 10 training labels: ", y_train[:10])
print("First 10 test labels: ", y_test[:10])

# Flatten the images
image_vector_size = 11
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)

# Re-inspect data shapes
# 28 * 28 = 784
print("Training data shape: ", x_train.shape)  # (60000, 784) -- 60000 images, each a flat series of 784 pixels
print("Test data shape: ", x_test.shape)  # (10000, 784) -- 10000 images, a flat series of 784 pixels

# Convert to "one-hot" vectors using the to_categorical function
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("First 5 training lables as one-hot encoded vectors:\n", y_train[:5])


# Define a neural network model
image_size = 11
num_classes = 10

model = Sequential()  # Documentation: https://keras.io/models/sequential/

# The input layer requires the special input_shape parameter which should match
# the shape of our training data.
model.add(Dense(units=128, activation='sigmoid', input_shape=(image_size,)))  # Dense = fully connected layers
# model.add(Dense(units=128, activation='sigmoid', input_shape=(256,)))  # Dense = fully connected layers
# model.add(Dense(units=32, activation='sigmoid', input_shape=(32,)))  # Dense = fully connected layers

model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model and keep track of progress
history = model.fit(x_train, y_train, batch_size=32, epochs=40, verbose=False, validation_split=.2)
# Evaluate the model
loss, accuracy  = model.evaluate(x_test, y_test, verbose=False)
# Display the results
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

print('Final test loss:', loss)
print('Final test accuracy:', accuracy)