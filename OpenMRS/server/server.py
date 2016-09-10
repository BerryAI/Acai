from flask import Flask, request, render_template, send_from_directory
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
import pickle
from base64 import b64encode, b64decode
import json

from os import path
import sys
CMD = path.dirname(path.realpath(__file__))
sys.path.append(path.join(CMD, '../..'))
import OpenMRS as om


app = Flask(__name__, static_url_path='')
app.debug = True

# Make sure: Service is stateless.

def decode_object(pickled):
    return pickle.loads(b64decode(pickled))

def encode_object(obj):
    return base64encode(pickle.dumps(obj))

@app.route('/recommend', methods=['POST'])
def handle_recommend():
    data = request.get_json(force=True)
    user_model = decode_object(data['user_model'])
    hidden_features = data['hidden_features']
    num_tracks = data.get('num_tracks', 2)
    return json.dumps(
        om.recommend_next_n_tracks(user_model, hidden_features, num_tracks)
    )

@app.route('/train_user_model', methods=['POST']):
def handle_train_user_model():
    data = request.get_json(force=True)
    ratings_one_user = data['ratings_one_user']
    hidden_features = data['hidden_features']
    user_model = om.train_user_taste_model(ratings_one_user, hidden_features)
    # Note: pickled_model is a base64 string. Only Acai understands how
    #   to use it.
    pickled_model = encode_object(user_model)
    return json.dumps({
        'user_model': pickled_model
    })

@app.route('/train_cf', methods=['POST']):
    data = request.get_json(force=True)
    all_ratings = decode_object(data['all_ratings'])
    return json.dumps(om.train_cf(all_ratings))
