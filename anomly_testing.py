import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,concatenate,Input,UpSampling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
model = load_model('auto-plant.h5')
img=cv2.imread("/Users/aadhyakaul/PycharmProjects/Hitachi/pre-processed2/Arjun (P1)/0013_0001.JPG")
img=cv2.resize(img,(256,256))
i=tf.reshape(img,[1,256,256,3])
img_reconstructed=model.predict(i)
print(img_reconstructed.shape)
loss = np.mean(np.abs(img_reconstructed - i), axis=1)
threshold = np.max(loss)
print(threshold)
# if loss.any() > 166:
#     print("anomly")
# else:
#     print("not anomly")
#
# cv2.imshow("image",img_reconstructed[0])
# cv2.waitKey()
# cv2.destroyAllWindows()
# loss = np.mean(np.abs(img_reconstructed - i), axis=1)

