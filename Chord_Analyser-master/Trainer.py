import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks


'''This code is for creating the model'''


image_x, image_y = 200, 200
batch_size = 64 #he number of samples processed during training
train_dir = "chords"


#creating CNN model architechture
def keras_model(image_x, image_y):
    num_of_classes = 5
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(image_x, image_y, 1), activation='relu')) #performing convolution on the input image
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')) #reducing the size of the feature maps obtained from previous layer 
    model.add(Conv2D(64, (5, 5), activation='relu')) #adding another convolution layer to extract more complex features
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same')) #adding another pooling layer to reduce the size of feature maps
    model.add(Flatten()) #flattening the previous layer into a one-dimensional vector
    model.add(Dense(1024, activation='relu')) #a fully connected layer with 1024 units
    model.add(Dropout(0.6)) #randomly drops some of the neurons in previous layer to prevent overfitting
    model.add(Dense(num_of_classes, activation='softmax')) #output layer with softmax activation which genrates output probabilities
    
    #compiling the modal and saving the best model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    filepath = "guitar_learner_final.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list


def main():
    #setting up the dataset for creating training and testing sets
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        rotation_range=15,
        zoom_range=0.2,
        horizontal_flip=False,
        validation_split=0.2,
        fill_mode='nearest')
    
    
    #Training dataset 80%
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_x, image_y),
        color_mode="grayscale",
        batch_size=batch_size,
        seed=42,
        class_mode='categorical',
        subset="training")
    
    class_indices = train_generator.class_indices
    print(class_indices)
    
    #Testing dataset 20%
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_x, image_y),
        color_mode="grayscale",
        batch_size=batch_size,
        seed=42,
        class_mode='categorical',
        subset="validation")

    model, callbacks_list = keras_model(image_x, image_y)
    
    print(callbacks_list)
    #model.fit_generator(train_generator, epochs=5, validation_data=validation_generator)
    #early_stopping = callbacks.EarlyStopping(monitor="val_accuracy",mode = "max",patience=5,verbose=0,restore_best_weights=True)
    
    #starts training with 25 epochs
    history = model.fit_generator(train_generator, epochs=25, validation_data=validation_generator)
    #scores = model.evaluate_generator(generator=validation_generator, steps=64)
    
    #printing results
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
    scores = model.evaluate_generator(generator=validation_generator, steps=64)
    
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    
    model_json = model.to_json()
    with open("model_final.json", "w") as json_file:
        json_file.write(model_json)
    
    model.save('guitar_learner_final.h5')




main()


