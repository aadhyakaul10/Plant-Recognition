import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense,Input,UpSampling2D,Dropout
dataset = np.load("train-test-plant.npz")
x_train, x_test = dataset["arr_0"], dataset["arr_1"]
x_train= x_train/255
x_test = x_test/255
model = Sequential()
#Encoder
#Input Layer
model.add(Input((256,256,3)))
#layer1
model.add(Conv2D(16,(5,5),activation="LeakyReLU",padding="same"))

model.add(MaxPooling2D((2,2), padding="same"))
#layer2
model.add(Conv2D(16, (3,3), activation="LeakyReLU", padding="same"))
model.add(Dropout(rate=0.2))
model.add(MaxPooling2D((3,3), padding="same"))
#layer3
model.add(Conv2D(8, (3,3), activation="LeakyReLU", padding="same"))
model.add(Dropout(rate=0.2))
model.add(MaxPooling2D((2,2), padding="same"))

model.add(Dense(15, activation= "relu"))

#Decoder
#layer1
model.add(Conv2D(8,(3,3), activation="LeakyReLU", padding="same"))
model.add(Dropout(rate=0.2))
model.add(UpSampling2D((2,2)))
# layer2
model.add(Conv2D(16,(3,3),activation="LeakyReLU", padding="same"))
model.add(Dropout(rate=0.2))
model.add(UpSampling2D((3,3)))
#layer3
model.add(Conv2D(16, (5,5), activation="LeakyReLU"))
model.add(Dropout(rate=0.2))
model.add(UpSampling2D((2,2)))


#Output Layer
model.add(Conv2D(3, (2,2), activation="sigmoid", padding="same"))


print(model.summary())
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
model.fit(x_train,x_train,epochs=20,validation_split=0.15)
model.save("auto-plant_final.h5")
model.evaluate(x_test,x_test)


tf.keras.utils.plot_model(model,to_file="auto-plant_final.png", show_shapes= True, show_layer_names=True)
