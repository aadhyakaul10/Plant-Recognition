import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
# import seaborn as sns

model = load_model("/Users/aadhyakaul/PycharmProjects/Hitachi/auto-plant_final.h5")
img = "/Users/aadhyakaul/PycharmProjects/Hitachi/pre-processed/Jamun (P5)/0005_0001.JPG"
img = cv2.imread(img)
img = cv2.resize(img, (256,256))
img = img/255
img = tf.reshape(img, [1,256,256,3])
img_reconstructed = model.predict(img)

loss = np.mean(np.abs(img_reconstructed - img), axis=1)
threshold = np.max(loss)
print("Threshold :", threshold)
if threshold > 0.225:
    print("anamoly")
else:
    print("not-anamoly")

plt.imshow(img_reconstructed[0])
plt.show()



