import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import os

SAVED_MODEL_FILE = 'saved_model.h5'

#Read data
full_data = pd.read_csv("train.csv")
print(full_data.shape)

def plot_training_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure()
    plt.plot(epochs, acc, 'b-')
    plt.plot(epochs, val_acc, 'b')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc = 'best')
    plt.show()

#Print 10 random number
fig, ax = plt.subplots(nrows = 5, ncols = 2, figsize = (15, 5))
for i, pic_number  in enumerate(np.random.randint(42000, size = 10)):
    #Convert data into array and strip the first value corresponding to the label
    label, image_array = np.split(np.asarray(full_data.loc[pic_number]), [1]) #Check if loc shouldn't be used to select rows by integer
    image_array = image_array.reshape((28, 28)) #Convert to image format

    ax[i // 2, i % 2].imshow(image_array, cmap = 'gray')
    ax[i // 2, i % 2].set_title('Label: ' + str(label[0]))
fig.subplots_adjust(wspace = 1, hspace = 1)
plt.show()

labels_original = np.asarray(full_data['label'])
images_original = np.asarray(full_data.loc[:, full_data.columns != 'label'])
images_original = images_original.reshape(images_original.shape[0], 28, 28, 1)

#Data augmentation
data_generator = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')



X_train, X_test, y_train, y_test = train_test_split(images_original, labels_original, test_size = 0.05) 

#Convert labels to categorical values 
#to_categorical is a one-hot conversion. The data must be converted to a numpy array beforehand in order to work well
y_train_converted = to_categorical(y_train, num_classes = 10)
y_test_converted  = to_categorical(y_test, num_classes = 10)

#Load model if it exists, or create one from scratch
if os.path.isfile(SAVED_MODEL_FILE):
    model = load_model(SAVED_MODEL_FILE)
    print('Model loaded')
else:
    #Create model
    model = Sequential()

    #Add model layers
    model.add(Conv2D(64, kernel_size = 3, activation = 'relu', input_shape = (28, 28, 1)))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(10, activation = 'softmax'))

    #Compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #Train the model
    #model.fit(X_train, y_train_converted, validation_data=(X_test, y_test_converted), epochs=5)
    data_generator.fit(X_train)
    image_iterator = data_generator.flow(X_train, y_train_converted)
    validation_iterator = data_generator.flow(X_test, y_test_converted)
    history = model.fit_generator(image_iterator, steps_per_epoch = 1000, epochs = 20, validation_data = validation_iterator)
    
    #Plot training history
    plot_training_history(history)

    #Save full model to HDF5 file (architecture, weight, etc)
    model.save(SAVED_MODEL_FILE)
    print('Model saved to disk')

#Make predictions
test_data = np.asarray(pd.read_csv('test.csv'))
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)
predictions = model.predict(test_data)

label_prediction = np.argmax(predictions, axis = 1)

image_id_list = []
label_list = []

for i, value in enumerate(label_prediction):
    image_id_list.append(i + 1)
    label_list.append(value)
    
#Convert to Dataframe
output_dict = {'ImageId': image_id_list, 'Label': label_list}
output_dataframe = pd.DataFrame(output_dict)

#Save results as csv file
output_dataframe.to_csv('predictions.csv', index = False)


