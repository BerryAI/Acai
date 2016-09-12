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
sys.path.append(path.join(CMD, '..'))
import OpenMRS as om


app = Flask(__name__, static_url_path='')
app.debug = True

# Make sure: Service is stateless.


def decode_object(pickled):
    return pickle.loads(b64decode(pickled))


def encode_object(obj):
    return b64encode(pickle.dumps(obj))


@app.route('/recommend', methods=['POST'])
def handle_recommend():
    data = request.get_json(force=True)
    if data.get('user_model') is None or data.get('hidden_features') is None:
        return json.dumps(
            {'error': 'Please provide user_model and hidden_features.'})
    user_model = decode_object(data['user_model'])
    hidden_features = data['hidden_features']
    for track_id in hidden_features:
        hidden_features[track_id] = decode_object(hidden_features[track_id])
    num_tracks = data.get('num_tracks', 2)
    tracks = om.recommend_by_user_model(user_model, hidden_features, num_tracks)
    return json.dumps(
        [t.to_dict() for t in tracks]
    )


@app.route('/train_user_model', methods=['POST'])
def handle_train_user_model():
    data = request.get_json(force=True)
    if (data.get('ratings_one_user') is None or
       data.get('hidden_features') is None):
        return json.dumps(
            {'error': 'Please provide ratings_one_user and hidden_features.'})
    ratings_one_user = data['ratings_one_user']
    hidden_features = data['hidden_features']
    for track_id in hidden_features:
        hidden_features[track_id] = decode_object(hidden_features[track_id])
    user_model = om.train_user_model(ratings_one_user, hidden_features)
    # Note: pickled_model is a base64 string. Only Acai understands how
    #   to use it.
    pickled_model = encode_object(user_model)
    return json.dumps({
        'user_model': pickled_model
    })


@app.route('/train_cf', methods=['POST'])
def handle_train_cf():
    data = request.get_json(force=True)
    if data.get('all_ratings') is None:
        return json.dumps({'error': 'Please provide all_ratings.'})
    all_ratings = data['all_ratings']
    hidden_features = om.train_cf(all_ratings)
    for track_id in hidden_features:
        hidden_features[track_id] = encode_object(hidden_features[track_id])
    return json.dumps(hidden_features)


if __name__ == '__main__':
    # app.run(port=5001)
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(80)
    IOLoop.instance().start()
