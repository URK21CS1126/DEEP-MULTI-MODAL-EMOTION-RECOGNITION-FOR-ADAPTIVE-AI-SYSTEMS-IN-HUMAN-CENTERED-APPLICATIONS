from flask import Flask,render_template,flash,redirect,request,send_from_directory,url_for, send_file,send_from_directory
import mysql.connector, os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
import torch
import nltk
# Download NLTK resources
nltk.download('stopwords')
import matplotlib.pyplot as plt
import wave
import pyaudio
from keras.models import load_model
import os
import librosa
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from audio_wave import *
import joblib

emotion_labels_txt = ['joy', 'sadness', 'anger', 'fear', 'love','surprise']

new_model = load_model(r"model/modecnnl.h5")
def preprocess_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    # Perform preprocessing steps (tokenization, lowercase conversion, etc.)
    preprocessed_data = [sample.strip().lower() for sample in data]
    return preprocessed_data

# Load and preprocess data
train_data = preprocess_data('train.txt')
test_data = preprocess_data('test.txt')
valid_data = preprocess_data('val.txt')

# Split data into features (text) and labels (emotions)
X_train, y_train = [sample.split(';')[0] for sample in train_data], [sample.split(';')[1] for sample in train_data]
X_test, y_test = [sample.split(';')[0] for sample in test_data], [sample.split(';')[1] for sample in test_data]

# Feature extraction
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vectorized, y_train)

# Load the model
loaded_model = joblib.load('bert_model.pkl')

# Prediction example for single input text
def predict_emotion(input_text):
    input_text_vectorized = vectorizer.transform([input_text])
    predicted_emotion = loaded_model.predict(input_text_vectorized)
    return predicted_emotion[0]

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='multi_emotion'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (email, password) VALUES (%s, %s)"
                values = (email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered! Please go to login section")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return redirect("/home")
            return render_template('home.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')


# @app.route('/image', methods=['GET', 'POST'])
# def image():
#     if request.method == 'POST':
#         myfile=request.files['file']
#         fn=myfile.filename
#         mypath=os.path.join('static/images', fn)
#         myfile.save(mypath)
#         accepted_formated=['jpg','png','jpeg','jfif']
#         if fn.split('.')[-1] not in accepted_formated:
#             flash("Image formats only Accepted","Danger")
#         new_model = load_model(r"model/modecnnl.h5")
#         test_image = image.load_img(mypath, target_size=(224, 224))
#         test_image = image.img_to_array(test_image)
#         test_image = test_image/255
#         test_image = np.expand_dims(test_image, axis=0)
#         result = new_model.predict(test_image)
#         print(result)
#         print(np.argmax(result))
#         classes=['Anger','Disgust','Fear',
#             'Happy','Neutral','Sad',
#             'Surprise']
#         prediction=classes[np.argmax(result)]
#         return render_template('image.html', prediction = prediction, path = mypath)
#     return render_template('image.html')

# Load the processor and model
processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")

@app.route('/upload', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':
        myfile=request.files['file']
        fn=myfile.filename
        mypath=os.path.join('static/images/', fn)
        myfile.save(mypath)
        accepted_formated=['jpg','png','jpeg','jfif']
        if fn.split('.')[-1] not in accepted_formated:
            flash("Image formats only Accepted","Danger")
        new_model = load_model(r"model/FinalModel.h5")
        test_image = image.load_img(mypath, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image/255
        test_image = np.expand_dims(test_image, axis=0)
        result = new_model.predict(test_image)
        print(result)
        print(np.argmax(result))
        classes=['Anger','Disgust','Fear',
            'Happy','Neutral','Sad',
            'Surprise'
            ]
        prediction=classes[np.argmax(result)]

        print(prediction)
        return render_template('image.html', prediction=prediction, path=mypath)

    return render_template('image.html')

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("static/images", filename)


CHUNK = 1024*4
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000

model=load_model("model/audio_model.h5")

#Extract features
def extract_features(file_name):
    X, sample_rate = librosa.load(file_name)
    #Short time fourier transformation
    stft = np.abs(librosa.stft(X))
    #Mel Frequency Cepstra coeff (40 vectors)
    mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    #Chromogram or power spectrum (12 vectors)
    chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    #mel scaled spectogram (128 vectors)
    mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    # Spectral contrast (7 vectors)
    contrast=np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    #tonal centroid features (6 vectors)
    tonnetz=np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz

#generating predictions
def speech_to_emotion(filename):
    mfccs, chroma, mel, contrast, tonnetz= extract_features(filename)

    features=np.empty((0,193))
    f=np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    features=np.vstack([features, f])

    his=model.predict(features)

    emotions=['neutral','calm','happy','sad','angry','fearful','disgused','surprised']
    y_pred=np.argmax(his, axis=1)
    pred_prob=np.max(his,axis=1)
    pred_emo=(emotions[y_pred[0]],pred_prob[0])

    


    return pred_emo

def record_audio(record=True, file_loc=None):
    if record:
        p=pyaudio.PyAudio()
        stream=p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK)

        # create matplotlib figure and axes
        fig, ax = plt.subplots(1, figsize=(7, 4))
        # variable for plotting
        x = np.arange(0, 2 * CHUNK, 2)
        # create a line object with random data
        line = ax.plot(x, np.random.rand(CHUNK), '-', lw=2)[0]
        # basic formatting for the axes
        ax.set_title('AUDIO WAVEFORM')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Volume')
        ax.set_xlim(0, 2 * CHUNK)
        plt.setp(ax, xticks=[0, CHUNK, 2 * CHUNK], yticks=[-1000, 1000])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # show the plot
        plt.show(block=False)
        wm = plt.get_current_fig_manager()
        wm.window.attributes('-topmost', 1)
        #wm.window.attributes('-topmost', 0)

        frames = []
        for i in range(0, int(RATE / CHUNK * 5)):
            data = stream.read(CHUNK)
            frames.append(data)
            result = np.frombuffer(data, dtype=np.int16)
            line.set_ydata(result)
            fig.canvas.draw()
            fig.canvas.flush_events()
            prog=round((i*100)/(int(RATE/CHUNK*5)))
            plt.suptitle('Progress: '+str(prog)+"%")

        filename = "output.wav"

        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        main_dir = r'output.wav'
        pred = speech_to_emotion(main_dir)
        #fig.suptitle(pred[0])
        fig.texts=[]
        plt.title('AUDIO WAVEFORM\nPredicted Emotion: '+str(pred[0].capitalize()))

        rec=wave.open(filename, 'r')
        return rec
    else:
        if file_loc:
            emo=speech_to_emotion(file_loc)[0]
            #print("The predicted emotion is: "+emo.capitalize())
            return emo.capitalize()


@app.route('/audio', methods=['GET', 'POST'])
def audio():
    if request.method == 'POST':
        myfile = request.files['file']
        fn = myfile.filename
        accepted_formats = ['mp3', 'wav', 'ogg', 'flac']
        if fn.split('.')[-1].lower() not in accepted_formats:
            message = "Invalid file format. Accepted formats: {}".format(', '.join(accepted_formats))
            return render_template("audio.html", message = message)
        mypath = os.path.join('static/audio/', fn)
        myfile.save(mypath)

        result = record_audio(record=False, file_loc=mypath) 
        print(111111111111111111111111111, result)
        return render_template('audio.html', prediction = result, path = mypath)
    return render_template('audio.html')


@app.route('/text', methods=['GET', 'POST'])
def text():
    if request.method == 'POST':
        user_input = request.form['file']
        predicted_emotion = predict_emotion(user_input)
                
        return render_template('text.html', prediction = predicted_emotion, user_input = user_input)
    return render_template('text.html')




if __name__ == '__main__':
    app.run(debug = True)