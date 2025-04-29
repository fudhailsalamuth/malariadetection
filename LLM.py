import pathlib
import os
import PIL
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Set the data directory path
data_dir = pathlib.Path("C:\\Users\\fudha\\OneDrive\\Desktop\\Malaria detection\\cell_images\\cell_images")
model_path = "malaria_cnn_model.h5"

# Function to build the CNN model
def build_model():
    input = Input(shape=(150, 150, 3), dtype=tf.float32, name="malaria_cells")
    X = Conv2D(64, (3, 3), padding="same", activation="relu")(input)
    X = MaxPooling2D((2, 2))(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)
    X = Conv2D(64, (3, 3), padding="same", activation="relu")(X)
    X = MaxPooling2D((2, 2))(X)
    X = GlobalAveragePooling2D()(X)
    X = Dense(512, activation="relu")(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)
    X = Dense(256, activation="relu")(X)
    X = BatchNormalization()(X)
    output = Dense(1, activation="sigmoid")(X)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
    return model

# Function to train the model
def train_model():
    batch_size = 32
    img_height = 150
    img_width = 150

    # Data Augmentation
    image_gen = ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.3,
        rescale=1 / 255
    )

    # Training and validation data generators
    training_data = image_gen.flow_from_directory(
        data_dir,
        subset="training",
        class_mode="binary",
        target_size=(img_width, img_height),
        batch_size=batch_size
    )

    validation_data = image_gen.flow_from_directory(
        data_dir,
        subset="validation",
        class_mode="binary",
        target_size=(img_width, img_height),
        batch_size=batch_size
    )

    model = build_model()

    # Training callbacks
    epochs = 5
    checkpoint_filepath = '/tmp/checkpoint.weights.h5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, min_delta=1e-3, restore_best_weights=True)

    # Train the model
    history = model.fit(training_data, epochs=epochs, validation_data=validation_data,
                        callbacks=[model_checkpoint_callback, early_stopping])

    # Save the model after training
    model.save(model_path)
    return model

# Function to evaluate an image using the trained model
def evaluate_image(image_path, model):
    image = load_img(image_path, target_size=(150, 150))
    img_arr = img_to_array(image) / 255.0
    pred = model.predict(np.expand_dims(img_arr, axis=0)).flatten()
    label = "Parasitized" if pred < 0.5 else "Uninfected"
    print(f"Prediction: {label} ({pred[0]:.2%} confidence)")

# Check if the model already exists
if os.path.exists(model_path):
    retrain = input("Model found! Do you want to retrain the model? (yes/no): ").lower()

    if retrain == 'yes':
        print("Retraining the model...")
        model = train_model()
    else:
        print("Loading the pre-trained model...")
        model = load_model(model_path)
else:
    print("No pre-trained model found. Training the model for the first time...")
    model = train_model()

# Input prompt for the user to test the model on an image
while True:
    image_path = input("Enter the path to the image you'd like to test (or type 'exit' to quit): ")
    if image_path.lower() == 'exit':
        break
    if os.path.exists(image_path):
        evaluate_image(image_path, model)
    else:
        print("Invalid image path. Please try again.")
