import tensorflow as tf
import cv2
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,concatenate,Input,UpSampling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True,rotation_range=90)
train_generator = train_datagen.flow_from_directory(directory="/Users/aadhyakaul/PycharmProjects/Hitachi/pre-processed",target_size=(256,256),class_mode="categorical")
model2 = Sequential()
model2.add(Conv2D(32, (5,5),activation="relu", input_shape=(256,256,3),padding="valid"))
model2.add(MaxPooling2D((2,2)))
model2.add(Dropout(0.2))

model2.add(Conv2D(16,(3,3),activation="relu",padding="valid"))
model2.add(MaxPooling2D((2,2)))
model2.add(Dropout(0.2))

model2.add(Conv2D(16,(4,4),activation="relu", padding="valid"))
model2.add(MaxPooling2D((2,2)))
model2.add(Dropout(0.2))

model2.add(Conv2D(8,(3,3),activation="relu", padding="valid"))
model2.add(MaxPooling2D((2,2)))
model2.add(Dropout(0.2))

model2.add(Flatten())
model2.add(Dense(20, activation="relu"))
model2.add(Dense(11, activation="softmax"))
print(model2.summary())
model2.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
model2.fit(train_generator,epochs=20)
model2.save("class-plant3_final.h5")
print(train_generator.class_indices.keys())

