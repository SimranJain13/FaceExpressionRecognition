from flask import Flask, url_for, send_from_directory, request, Response
import logging, os
from werkzeug.utils import secure_filename
import json
import io
from PIL import Image
import cv2
import numpy as np
import pytesseract
#import Skewness
import string
import facial_emotion_image


import tensorflow as tf
import cv2 as cv
import imutils
from operator import itemgetter


import shutil
import csv
import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.preprocessing.image import img_to_array
import pickle
import string, random

#count=0
app = Flask(__name__)
#app.config['MAX_CONTENT_LENGTH'] = 1* 1024 * 1024  # for 1MB max-limit.
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
# UPLOAD_ORG_FOLDER = '{}original/'.format(UPLOAD_FOLDER)
# UPLOAD_PREP_FOLDER = '{}preprocessed/'.format(UPLOAD_FOLDER)
#cpt = sum([len(files) for r, d, files in os.walk(UPLOAD_FOLDER)])
#count=cpt+1;
#print("files:",count);
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['UPLOAD_PREP_FOLDER'] = UPLOAD_PREP_FOLDER

def create_new_folder(local_dir):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    return local_dir



@app.route('/detectface', methods = ['POST'])
def api_root():
    #app.logger.info(PROJECT_HOME)
    #app.logger.info(UPLOAD_FOLDER)
    if request.method == 'POST' and request.files['image']:

        data = request.files.to_dict()
        inner_data = data['image']
        format = str(inner_data)
        i = format.find('image/jpeg')
        j = format.find('image/png')
        img = request.files['image'].read()
        size = len(img)
        cpt = sum([len(files) for r, d, files in os.walk(UPLOAD_FOLDER)])
        count = cpt + 1

        if i != -1 or j != -1:
            if size <= 1048576:
                img = request.files['image']
                original_img = Image.open(img)
                # resized_img = original_img.resize((320, 320))
                img_name = secure_filename(img.filename)
                #print("image name:", img_name)
                #output = ocr_core(img_name)
                sub = img_name.rfind('.')
                length = len(img_name)
                substr = img_name[sub:length]


                create_new_folder(app.config['UPLOAD_FOLDER'])
                count = cpt + 1
                saved_path = os.path.join(app.config['UPLOAD_FOLDER'], str(count) + substr)

                app.logger.info("saving {}".format(saved_path))
                original_img.save(saved_path)
                #print(saved_path)
                recognised_text = facial_emotion_image.MainFunction(saved_path)
                #recognised_text = recognised_text.upper()
                print(recognised_text)
                if recognised_text is not None and len(recognised_text) > 0 :
                    return Response(json.dumps({"emotion_probability": recognised_text}), mimetype='application/json') #return(recognised_text)
                else:   return Response(json.dumps({"emotion_probability":recognised_text}), mimetype='application/json') #return("success")
            else:
                    return Response(json.dumps({"emotion_probability": "Size should not be greater than 1 mb"}),mimetype='application/json')  #return("success")return "Size should not be greater than 1 mb"
        else:
            return Response(json.dumps({"emotion_probability": "Size should not be greater than 1 mb"}), mimetype='application/json')  # return("success")return "Size should not be greater than 1 mb"
    else:
        return Response(json.dumps({"emotion_probability": "Where is the image?"}),mimetype='application/json')  # return("success")return "Size should not be greater than 1 mb"

if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=False)
