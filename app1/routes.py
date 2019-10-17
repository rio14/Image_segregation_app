from flask import render_template, url_for, flash, redirect, request,jsonify
from app1 import app, db, bcrypt
from app1.forms import RegistrationForm, LoginForm
from app1.models import User
from flask_login import login_user, current_user, logout_user, login_required
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
import pandas as pd
import tensorflow as tf

from numpy import savez_compressed
from numpy import load
from numpy import expand_dims
from numpy import asarray

from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

from random import choice
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import pickle

import os
import stripe

model = load_model('facenet_keras.h5')
global graph
graph = tf.get_default_graph()
print('Loaded Model use: http://127.0.0.1:5000/')

stripe_keys = {
  'secret_key': 'sk_test_gCOlmojuFSA7ClIgDFv2BeZZ00ip4F8ZXc',
  'publishable_key':'pk_test_8KCgEPgZFt6wMfRAmM2GV8Pr00GE2vtmHI'
}

stripe.api_key = stripe_keys['secret_key']

def extract_face(filename, required_size=(160, 160)):
    # load image from file
    #image1 = Image.open(filename)
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

# load images and extract faces for all images in a directory
def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        face = extract_face(path)
        # store
        faces.append(face)
    return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

# get the face embedding for one face
def get_embedding(model, face_pixels):
    with graph.as_default():
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        return yhat[0]


@app.route('/charge', methods=['POST'])
def charge():
  amount = 500

  stripe.Charge.create(
    amount=amount,
    currency='usd',
    card=request.form['stripeToken'],
    description='Stripe Flask'
  )

  return render_template('home.html', amount=amount)

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html' ,key=stripe_keys['publishable_key'])
required_size=(160, 160)

@app.route("/about")
@login_required
def about():
    return render_template('about.html', title='About')

get_name='target'

@app.route('/submit',methods =['GET','POST'])
def submit():
    if current_user.is_authenticated:

        folder_path = request.form['loc_i']
        folder_path2 = request.form['loc_r']
        folder_path3 = request.form['loc_o']
        if folder_path:
            trainX, trainy = load_dataset(folder_path)
            print("data loading complete",'success')
            #convert each face in the train set to an embedding
            newTrainX = list()
            for face_pixels in trainX:
                embedding = get_embedding(model, face_pixels)
                newTrainX.append(embedding)
            newTrainX = asarray(newTrainX)
            #print(newTrainX.shape)
            in_encoder = Normalizer(norm='l2')
            trainX = in_encoder.transform(newTrainX)
            # label encode targets
            out_encoder = LabelEncoder()
            out_encoder.fit(trainy)
            trainy = out_encoder.transform(trainy)
            # fit model
            model2 = SVC(kernel='linear', probability=True)
            model2.fit(trainX, trainy)
            pickle.dump(model2, open(str(folder_path)+"ml_model.sav", 'wb'))
            print("Model training complete",'success')

            print("path recieved")
            model1 = pickle.load(open(str(folder_path)+"ml_model.sav", 'rb'))
            print('trained model loaded')
            for file in listdir(folder_path2):
                image1 = Image.open(folder_path2+file)
            # convert to RGB, if needed
                image = image1.convert('RGB')
            # convert to array
                pixels = asarray(image)
            # create the detector, using default weights
                detector = MTCNN()
            # detect faces in the image
                results = detector.detect_faces(pixels)
            # extract the bounding box from the first face
                x1, y1, width, height = results[0]['box']
            # bug fix
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
            # extract the face
                face = pixels[y1:y2, x1:x2]
            # resize pixels to the model size
                image = Image.fromarray(face)
                image = image.resize(required_size)
                face_array = asarray(image)
                faces = list()
                faces.append(face_array)
                X, y = list(), list()
                X.extend(faces)
                X = asarray(X)

                newTrainX = list()
                embedding = get_embedding(model, X[0])
                newTrainX.append(embedding)
                newTrainX = asarray(newTrainX)
                print(newTrainX.shape)

                in_encoder = Normalizer(norm='l2')
            #trainX1 = in_encoder.transform(trainX1)
                X1 = in_encoder.transform(newTrainX)
                random_face_emb = X1[0]
            # prediction for the face
                samples = expand_dims(random_face_emb, axis=0)
                yhat_class = model1.predict(samples)
                yhat_prob = model1.predict_proba(samples)
            # get name
                class_index = yhat_class[0]
                class_probability = yhat_prob[0,class_index] * 100
                names = out_encoder.inverse_transform(yhat_class)
                print(names)
                if str(names[0]) == get_name:
                    image1.save(folder_path3+file)

            return render_template('thankyou.html')
        return jsonify({'error': 'Missing Data!'})
    else:
        return redirect(url_for('login'))




@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route("/account")
@login_required
def account():
    return render_template('account.html', title='Account')
