from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, Conv1D, MaxPooling1D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
# from IPython.display import display
from PIL import Image
import h5py

# CNN model
from scripts.machineLearning.ml_data_for_classification import MlDataForModelTraining, MlDataForModelTesting

classifier = Sequential()

classifier.add(Conv1D(32, 2, input_shape=(11,), activation='relu'))
classifier.add(MaxPooling1D(pool_size=2))
classifier.add(Conv1D(32,  3, activation='relu'))
classifier.add(MaxPooling1D(pool_size=2))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=10, activation='softmax'))  # number of classes

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation
batch_size = 32
# train_datagen = ImageDataGenerator(rescale = 1./255,
#                                    shear_range = 0.2,
#                                    zoom_range = 0.2,
#                                    rotation_range = 20,
#                                    horizontal_flip = True)
#
# test_datagen = ImageDataGenerator(rescale = 1./255)
#
# training_set = train_datagen.flow_from_directory('training_set',
#                                                  target_size = (64, 64),
#                                                  batch_size = batch_size,
#                                                  class_mode = 'categorical')
#
# test_set = test_datagen.flow_from_directory('test_set',
#                                             target_size = (64, 64),
#                                             batch_size = batch_size,
#                                             class_mode = 'categorical')

train_data_angles = MlDataForModelTraining(
    '/csv_data_files/train_data_angles_with_flipped.csv')
test_data = MlDataForModelTesting(
    '/csv_data_files/test_data_angles.csv')

classifier.fit(train_data_angles.train_data,
                # number of training set images, 729
               epochs=1,
             )  # number of test set images, 229

classifier.save('my_model_multiclass10.h5')  # save model
