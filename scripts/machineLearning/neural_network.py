import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
import random
from scripts.machineLearning.ml_data_for_classification import MlDataForModelTraining, MlDataForModelTesting

train_data_angles = MlDataForModelTraining(
    '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/experiments/own_split_test/csv_data_files/train_data_angles_with_flipped.csv')
test_data_angles = MlDataForModelTesting('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/experiments/own_split_test/csv_data_files/test_data_angles.csv')

# Inspect train data
print("Training data shape: ", train_data_angles.train_data.shape)

# Inspect labels
print("Training labels shape: ", train_data_angles.train_labels.shape)  # (60000, 28, 28) -- 60000 numbers from the range 0, 1, ... 9
print("Test labels shape: ", test_data_angles.test_labels.shape)  # (10000, 28, 28) -- 10000 numbers from the range 0, 1, ... 9
print("First 10 training labels: ", train_data_angles.train_labels[:10])
print("First 10 test labels: ", test_data_angles.test_labels[:10])

# Re-inspect data shapes
print("Training data shape: ", train_data_angles.train_data.shape)
print("Test data shape: ", test_data_angles.test_data.shape)


num_classes = 10

# random.shuffle(train_data_angles.train_labels)
# random.shuffle(train_data_angles.train_data)
# random.shuffle(test_data_angles.test_labels)
# random.shuffle(test_data_angles.test_data)

# Convert to "one-hot" vectors using the to_categorical function
y_train = keras.utils.to_categorical(train_data_angles.train_labels, num_classes)
print("First 10 test labels: ", test_data_angles.test_labels[:10])

y_test = keras.utils.to_categorical(test_data_angles.test_labels, num_classes)

# Re-inspect train labels
print("First 5 training lables as one-hot encoded vectors:\n", y_train[:5])
# Re-inspect test labels
print("First 5 test lables as one-hot encoded vectors:\n", y_test[:5])


# Define a neural network model
num_features = 11
model = Sequential()  # Documentation: https://keras.io/models/sequential/

# The input layer requires the special input_shape parameter which should match
# the shape of our training data.
model.add(Dense(units=32, activation='sigmoid', input_shape=(num_features,)))  # Dense = fully connected layers
model.add(Dense(units=32, activation='sigmoid'))  # Dense = fully connected layers
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model and keep track of progress
history = model.fit(train_data_angles.train_data, y_train, batch_size=16, epochs=4, verbose=False, validation_split=.2)

# Evaluate the model
loss, accuracy  = model.evaluate(test_data_angles.test_data, y_test, verbose=False)

print('Final test loss:', loss)
print('Final test accuracy:', accuracy)


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