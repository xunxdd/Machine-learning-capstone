from flask import Flask
from flask import render_template
from keras.models import load_model
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import os
from flask import request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
from keras.models import load_model
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

labels = ['Normal', 'Pneumonia - Bacterial', 'Pneumonia - Viral']

UPLOAD_FOLDER = '/uploadimgs/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = basedir + '/static/uploadimgs/'
MODEL_Path = basedir + '/model.h5'


app = Flask(__name__)

global model, InceptionResNetV2_model

model = load_model(MODEL_Path)
InceptionResNetV2_model = InceptionResNetV2(weights='imagenet', include_top=False)
graph = tf.get_default_graph()


def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(320, 320))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def extract_InceptionResV2(tensor):
    fea = preprocess_input(tensor)
    pred = InceptionResNetV2_model.predict(fea)
    return pred


def predict(img_path):
    tensor = path_to_tensor(img_path)

    # obtain predicted vector
    with graph.as_default():
        bottleneck_feature = extract_InceptionResV2(tensor)
        predicted_vector = model.predict(bottleneck_feature)
    return labels[np.argmax(predicted_vector)]


@app.route("/")
def index():
    return render_template("index.html")



@app.route('/api/upload', methods=["POST"])
def upload():

    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(img_path)
            label = predict(img_path)
            return jsonify({'file': filename, 'label': label})


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    # run the app locally on the givn port
    app.run(host='0.0.0.0', port=port, debug=True)
