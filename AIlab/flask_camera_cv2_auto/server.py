import numpy as np
from PIL import Image
import os, cv2, requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from function.converse_label import string_to_point
from function.show import show_point # need to tag 2 key point
import flask, sys

numClasses = 31 #categories
app = flask.Flask(__name__)
app.secret_key = "secret key"
model = None

camera_mode = True
global frame
frame = None
camera = cv2.VideoCapture(0)


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

def frames():
    
    while True:
        success, frame = camera.read()
        if success:
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass


@app.route('/')
def home():
    return flask.render_template('page.html')

@app.route('/video_feed')
def video_feed():
    global camera_mode
    if camera_mode == True:
        return flask.Response(frames(), mimetype = 'multipart/x-mixed-replace; boundary=frame')
    else:
        ret, buffer = cv2.imencode('.jpg', cv2.flip(show_pic, 1))
        show = buffer.tobytes()
        return flask.Response(show, mimetype = 'multipart/x-mixed-replace;')

@app.route("/setCamera", methods=["POST", 'GET'])
def setCamera():
    global camera, camera_mode
    if flask.request.method == 'POST':
        camera.release()
        cv2.destroyAllWindows()
        camera_mode = True
        camera = cv2.VideoCapture(0)
    elif flask.request.method=='GET':
        return flask.render_template('page.html')
    return flask.render_template('page.html')

@app.route("/predict", methods=["POST"])
def predict():
    global camera_mode, show_pic
    print("predict, camera_mode is", camera_mode)
    if flask.request.method == "POST":
        ret, img = camera.read()
        camera_mode = False
        camera.release()
        cv2.destroyAllWindows()
        test_data = []
        if ret:
            test_data.append(cv2.resize(np.array(img), (50, 50)))
            test_data = np.array(test_data)/255
            output = model.predict(test_data)
            output = np.argmax(output, axis = -1)
            ans = classes[output[0]]
            a, b, c = string_to_point(ans)
            try:
                show_pic = cv2.cvtColor(show_point(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB), a, b, c, grid_number = 5, const_rate = 8, width_rate = 2, whether_grid = False, show_ori = True, whether_show = False, lower_bound = 30, upper_bound = 170), cv2.COLOR_BGR2RGB)
                return flask.render_template('page.html', prediction_display_area='answer：{}'.format(ans))
            except:
                show_pic = img
                return flask.render_template('page.html', prediction_display_area='Some error happened')
        else:
            camera_mode = True
            return flask.render_template('page.html', prediction_display_area='not ret')
    return flask.render_template('page.html', prediction_display_area='not post')

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
    
    
    
    
camera.release()
cv2.destroyAllWindows()   