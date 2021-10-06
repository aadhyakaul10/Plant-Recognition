from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import os
model_encoder = load_model("/Users/aadhyakaul/PycharmProjects/Hitachi/auto-plant_final.h5")
model_classification = load_model("/Users/aadhyakaul/PycharmProjects/Hitachi/class-plant3_final.h5")
def preprocess(path):
    img = cv2.imread(path)
    img = img/255
    img = cv2.resize(img,(256,256))
    img = tf.reshape(img,[1,256,256,3])
    return img
def outcomes(img):
    img_reconstructed=model_encoder.predict(img)
    loss = np.mean(np.abs(img_reconstructed - img), axis=1)
    threshold = np.max(loss)
    name = model_classification.predict(img)
    name = name.round()
    dic1 = {0:"Devils Tree",1:"Arjun",2:"Basil",3:"Chinar",4:"Gauva",5:"Jamun",6:"Jatropha",7:"Lemon",8:"Mango",9:"Pomegranate",10:"Pongame oiltree"}
    dic2 = {0:"Alstonia Scholaris",1:"Terminalia arjuna",2:"Ocimum basilicum",3:"Platanus orientalis",4:"Psidium guajava",5:"Syzygium cumini",6:"Jatropha curcas",7:"Citrus × limon",8:"Mangifera indica",9:"Punica granatum",10:"Pongamia Pinnata"}
    name = list(name[0])
    if 1 not in name:
        common_name= "error"
        scientific_name="error"
    else:
        index = name.index(1)
        common_name = dic1[index]
        scientific_name = dic2[index]
    if threshold.any() > 225:
        outcome1= "diseased"
    else:
        outcome1 = "healthy"
    return outcome1 , common_name , scientific_name






app = Flask(__name__)
upload = "static"
app.config["upload"] = upload

@app.route("/",methods = ["GET","POST"])
def pred():
    if request.method == "GET":
        return render_template("index.html", okay ="",common_name="",scientific_name="", file = "tree.jpeg")
    else:
        a = request.files["img"]
        if not a:
            return render_template("server_error.html")
        images = os.listdir(app.config["upload"])
        if len(images) <= 2:
            a.save(os.path.join(app.config["upload"], a.filename))
        else:
            image = images[0]
            if image == "tree.jpeg":
                image = images[1]
            os.remove(app.config["upload"]+"/"+image)
            a.save(os.path.join(app.config["upload"], a.filename))

        img = preprocess(os.path.join(app.config["upload"], a.filename))
        result, common_name, scientific_name  = outcomes(img)
        if common_name == "error" and scientific_name == "error":
            return render_template("server_error.html")
        return render_template("index.html",file=a.filename, okay = result,common_name = common_name,scientific_name= scientific_name)
if __name__ == "__main__":
    app.run(port = "8000")
    