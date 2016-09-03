"""Run this script to retrieve user ratings from ACAI game and save it to file.
"""
import sys
from os import path
import json
CMD = path.dirname(path.realpath(__file__))
sys.path.append(path.join(CMD, '../read_game_data'))
from acai_game import get_user_rate_dict, get_track_info_by_trackID


# Download ratings.
ratings = get_user_rate_dict()
json.dump(ratings,
          open(path.join(CMD, '../data/acai_game_user_ratings.json'), 'w'))

# Download detailed track info.
ratings = json.load(open(path.join(CMD, '../data/acai_game_user_ratings.json')))
tracks = {}
for _, rating_per_user in ratings.iteritems():
    for track_id in rating_per_user:
        if not track_id in tracks:
            print 'getting track', track_id
            track_info = get_track_info_by_trackID(track_id)
            tracks[track_id] = track_info
json.dump(tracks, open(path.join(CMD, '../data/tracks.json'), 'w'))
