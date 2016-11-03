import json
from os import path
import sys
from OpenMRS.catalog import Track
CWD = path.dirname(path.realpath(__file__))


def get_example_ratings():
    """
    Returned `data` is a dict of the following structure:
       user_id_1: {
           track_id_1: rating_1,
           track_id_2: rating_2, ...
       } ...
    """
    data = json.load(open(path.join(CWD, 'acai_game_user_ratings.json')))
    data = dict(data.items()[:5]) # TODO: remove this!
    return data


def get_example_tracks(ratings=None):
    ratings = ratings or get_example_ratings()
    saved_tracks = json.load(open(path.join(CWD, 'tracks.json')))
    tracks = {}
    for _, rating_per_user in ratings.iteritems():
        for track_id in rating_per_user:
            track_data = saved_tracks.get(track_id) or {
                'id': track_id,
                'title':'Track %s' % track_id,
                'artist': 'Unknown',
                'streaming_url': None
            }
            tracks[track_id] = Track(track_data=track_data, source='example')
    return tracks.values()
