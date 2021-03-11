from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='_weights.08-0.01.hdf5'

# Load your trained model
model = load_model(MODEL_PATH)

upload_folder="/home/kpr/Music/deep learning/testing/uploads"

def predict(img_path,model):
    img=image.load_img(img_path,target_size=(200,200))
    

    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    images=np.vstack([x])
    result=model.predict(images)
    return result



@app.route('/', methods=['GET','POST'])
def upload_predict():
    # Main page
    if request.method=='POST':
        image_file=request.files["image"]
        if image_file:
            image_location=os.path.join(
                upload_folder,
                image_file.filename
            )
            image_file.save(image_location)
            pred=predict(image_location,model)
            ret=""
            if pred==0:
                ret+="Normal"
            else:
                ret+="Pneumonia"

            return render_template('index.html',prediction=ret)
    return render_template('index.html')    
    



if __name__ == '__main__':
    app.run(debug=True)
