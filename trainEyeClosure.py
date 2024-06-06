
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_SIZE = (100, 100)
BATCH_SIZE = 32
EPOCHS = 10


model = models.Sequential()
# model.add(layers.Conv2D(64, (3, 3), activation="relu", input_shape=(100, 100, 1)))
# model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

# model.add(layers.Conv2D(32, (3, 3), activation="relu"))
# model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

# model.add(layers.Conv2D(16, (3, 3), activation="relu"))
# model.add(layers.MaxPooling2D((2, 2)))

# model.add(layers.Conv2D(16, (3, 3), activation="relu"))
# model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))

model.add(layers.Dense(256, activation="relu"))
# model.add(layers.Dropout(0.2))
model.add(layers.Dropout(0.5))

# model.add(layers.Dense(64, activation="relu"))
# model.add(layers.Dropout(0.2))

model.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory("EyesData/train", target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, color_mode="grayscale", class_mode="binary")
test_generator = test_datagen.flow_from_directory("EyesData/test", target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, color_mode="grayscale", class_mode="binary")


history = model.fit(train_generator, epochs=EPOCHS, validation_data=test_generator)


model.summary()

model.save("eyes_closure_detection_model.h5")


