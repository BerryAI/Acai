"""Load a sample user rating data, train a recommendation engine,
and see what tracks get recommended.
"""

from os import path
import sys
CWD = path.dirname(path.realpath(__file__))
sys.path.append(path.join(CWD, '../..'))
# The above lines are not necessary when OpenMRS has already been installed.

import OpenMRS as om


example_ratings = om.data.get_example_ratings()
example_tracks = om.data.get_example_tracks()

engine = om.RecommendationEngine()  # or equivalently, use the following line
# engine = om.RecommendationEngine(catalog=SimpleCatalog(example_tracks))
engine.train(ratings=example_ratings)

one_user = engine.get_user_ids()[0]
ratings = engine.get_ratings_by_user(user_id=one_user)
print 'Ratings by user %s:' % one_user
for track_id, rating in ratings.items()[:5]:
    print '  User rates %s on track %s' % (rating,
        engine.catalog.get_track_by_id(track_id))

# Recommend tracks for a user.
recommended_tracks = engine.recommend(user_id=one_user, num=10)
print '\nRecommended tracks for user %s:' % one_user
for t in recommended_tracks:
    print t


# Advanced usage.

# Train a user model.
another_user = 'another_user'
her_ratings = {}
her_ratings[example_tracks[0].id] = 4
her_ratings[example_tracks[1].id] = 5
her_ratings[example_tracks[2].id] = 1
print '\nAnother user\'s ratings:'
for track_id, rating in ratings.items()[:5]:
    print '  User rates %s on track %s' % (rating,
        engine.catalog.get_track_by_id(track_id))
user_taste_model = engine.train_user_taste_model(her_ratings)
recommended_tracks = engine.recommend_by_user_model(user_taste_model)
print '\nRecommended tracks for another user:'
for t in recommended_tracks:
    print t

# Recommend tracks based on seed tracks.
# TODO: recommendation by seed tracks is not yet implemented.
# example_track_ids = [track.id for track in engine.get_tracks(offset=0, limit=2)]
# recommended_tracks = engine.recommend(seed_track_ids=example_track_ids, num=10)
# print recommended_tracks
