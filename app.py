
from flask import Flask, render_template, url_for,send_file, make_response, request
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
app = Flask(__name__)


def load_model():
    module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"
    #module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5"
    module = hub.load(module_handle)
    return module

def preprocess_img(img):
  img = tf.image.resize_with_pad(img,224,224)
  img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis,...]
  return img

@app.route('/get-features', methods=['POST'])
def get_image_vector():
    model = load_model()
    request_data = request.get_json()
    img = np.array(request_data['image'])
    img = preprocess_img(img)
    features = model(img)
    features = np.squeeze(features)
    return {"features" : features.tolist()}


@app.route('/hello', methods=['GET','POST'])
def hello():
    return "hello"


if __name__ == '__main__':
<<<<<<< HEAD
   app.run(debug = True)
=======
   app.run(debug = True )
>>>>>>> 572a1b020a7f3621279cc90f87f4bc02a7060024
