from io import BytesIO
import numpy as np
from PIL import Image
import os, cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
import requests
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from urllib.request import urlopen
from argparse import ArgumentParser
import sys
import flask
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
numClasses = 31
app = flask.Flask(__name__)
model = None
data_transforms = None

class CNN_model():
    def __init__(self ):
        super(CNN_model, self).__init__()
        
    
def load_data(cls_file, log_file):
    global classes, performance
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
    with open(log_file, "r") as f:
        performance = f.read().split("\n")

def load_model(path):
    global model
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = (50, 50, 3), activation = 'relu'))
    model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
    model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(numClasses, activation = 'sigmoid'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
    try:
        model.load_weights('model/model.h5')
        print("load successed")
    except:
        print("load failed")
    
@app.route('/')
def home():
    return flask.render_template('page.html')

@app.route("/predict", methods=["POST"])
def predict():
    output_dict = {"success": False}
    if flask.request.method == "POST":
        test_data = []
        img = flask.request.files['photo']
        img = Image.open(img.stream)
        test_data.append(cv2.resize(np.array(img), (50, 50)))
        #copy = np.array(img_to_array(img))/255.0
        test_data = np.array(test_data)/255
    return render_template('page.html',prediction_display_area='answer：{}'.format('1'))
        output = model.predict(test_data)
        output_dict["predictions"] = classes[output]
        output_dict["success"] = True
    return render_template('page.html',prediction_display_area='answer：{}'.format(output_dict["predictions"]))


@app.route("/performance", methods=["GET"])
def performance():

    output_dict = {"success": False}
    if flask.request.method == "GET":
        output_dict["performance"] = performance
        output_dict["success"] = True
    return flask.jsonify(output_dict), 200



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--eval', help='evaluation log', type=str, default=None)
    parser.add_argument('--model', help='model path', type=str, default=None)
    parser.add_argument('--port', help='port', type=int, default=5000)
    parser.add_argument('--classes', help='classes path', type=str, default=None)
    args = parser.parse_args()
    print(("* Loading model and Flask starting server... please wait until server has fully started"))
    if (args.model is None):
        print("You have to load the model by --model")
        sys.exit()
    if (args.classes is None):
        print("You have to load the classes file by --classes")
        sys.exit()
    if (args.eval is None):
        print("You have to load the evaluation log by --eval")
        sys.exit()
    load_model(args.model)
    load_data(args.classes, args.eval)
    app.run(host="0.0.0.0", debug=True, port=args.port)