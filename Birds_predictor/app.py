import numpy as np
import json
import os
import shutil
import config
from werkzeug.utils import secure_filename
from flask import Flask,render_template,redirect,request,url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
app = Flask(__name__)

#
folder = 'uploads'
# loading the classes using os module
with open (config.class_os_path,'rb') as f:
    classes = json.load(f)
model_load = load_model(config.model_os_path) #loading the model

#control the uploading file extension
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSION'] = set(['jpg','jpeg','png'])

#create function to control the uploading file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in app.config['ALLOWED_EXTENSION']

#creating home page API
@app.route("/")
def decorate_home():
    return render_template('home.html',redirect='/upload')


@app.route("/upload", methods =['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect('/predict')
    else:
        return 'No image was uploaded'
    

# converting  image to array 
def img_arry():
    # arranging the uploaded folder
    sorted_list = sorted(os.listdir(folder), key=lambda x: os.path.getmtime(os.path.join(folder, x)),reverse=True)
    if len(sorted_list)==0:
        print("folder is empty")
        files_name = 'test_image.jpeg'
    else:
        files_name = "".join(sorted_list[0])
    #joining file and uploaded file path
    pred_img_path = os.path.join(folder,files_name)
    img_pre_prcs = image.load_img(pred_img_path,target_size=(200,200)) #loading image
    inp_array = image.img_to_array(img_pre_prcs) # converting image to array
    img_to_arry = np.array([inp_array])
    img_arry = img_to_arry.astype('float32')/255
    return img_arry



#predicting the bird type
@app.route('/predict')
def predict():
    img_arr = img_arry()
    pred = model_load.predict([img_arr])
    pred_class = np.argmax(pred, axis=1) # return the number of class that bird belong
    for i,birds in enumerate(classes) :
        if i== pred_class[0]:
            print("Predicted class is : ",birds)
            return render_template('index.html',prediction = birds)
        

@app.route("/emty_fldr")
def empty_folder():
    for file in os.listdir(folder):
        if file == 'test_image.jpg':
            pass
        else:
            file_path = os.path.join(folder,file)
            os.remove(file_path)
    return render_template("empty.html")



if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080)
     
