import keras
from keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

from scripts.machineLearning.ml_data_for_classification import MlDataForModelTraining, MlDataForModelTesting


train_data = MlDataForModelTraining('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/old_split_run_through/csv_data_files/train_data_both_with_flipped.csv')
test_data = MlDataForModelTesting('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/old_split_run_through/csv_data_files/test_data_both.csv')

# Inspect data
print("Training data   shape: ", train_data.train_data.shape)
print("Training labels shape: ", train_data.train_labels.shape)
print("Test     data   shape: ", test_data.test_data.shape)
print("Test     labels shape: ", test_data.test_labels.shape)

# Inspect labels
print("First 10 training labels: ", train_data.train_labels[:10])
print("First 10 test     labels: ", test_data.test_labels[:10])

# Define a neural network model
feature_size = 21
num_classes = 10

# Convert to "one-hot" vectors using the to_categorical function
y_train = keras.utils.to_categorical(train_data.train_labels, num_classes)
y_test = keras.utils.to_categorical(test_data.test_labels, num_classes)

# Inspect one-hot vectors
print("First 10 training one-hot vectors: ", y_train[:10])
print("First 10 test     one-hot vectors: ", y_test[:10])

model = Sequential()  # Documentation: https://keras.io/models/sequential/

# The input layer requires the special input_shape parameter which should match
# the shape of our training data.
model.add(Dense(units=256, activation='relu', input_shape=(feature_size,)))  # Dense = fully connected layers
model.add(Dense(units=128, activation='relu'))  # Dense = fully connected layers
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer="adam", loss='kullback_leibler_divergence', metrics=['accuracy']) # default loss: categorical_crossentropy, can also be: kullback_leibler_divergence
model.summary()

# Train the model and keep track of progress
history = model.fit(train_data.train_data, y_train,
                    batch_size=32,
                    epochs=150,
                    verbose=1,
                    validation_split=0,
                    )

# Evaluate the model
loss, accuracy = model.evaluate(test_data.test_data, y_test, verbose=False)

print('Final test loss:', loss)
print('Final test accuracy:', accuracy)

model.save('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/neural_networks/og_split_both_test.h5')



#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.title('Accuracy')
#plt.xlabel('Epoch')
#plt.ylabel('Accuracy')
#plt.legend(['Train', 'Test'], loc='best')
#plt.show()

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('Loss')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.legend(['Train', 'Test'], loc='best')
#plt.show()
