"""Recommend API.
"""
from copy import deepcopy
import numpy as np
from catalog import SimpleCatalog, Track
from cf.cf_hidden_feature import get_hidden_feature_matrix_GD
import itertools
from API.user import train_user_taste_model

# TODO: implement classes: UserTasteModel.


class RecommendationEngine(object):

    def __init__(self, catalog=None):
        if catalog is None:
            import data
            catalog = SimpleCatalog(data.get_example_tracks())
        self.__catalog = catalog
        self._hidden_feature = {}

    @property
    def catalog(self):
        return self.__catalog

    @property
    def hidden_feature(self):
        return self._hidden_feature

    def train(self, ratings):
        k = 5
        learn_rate = 0.001
        lambda_rate = 0.04
        max_iter = 10000
        GD_method = 1
        user_weight, hidden_feature, res_norm, user_index, song_index = (
            get_hidden_feature_matrix_GD(
                ratings, k, learn_rate, lambda_rate, max_iter, GD_method)
        )
        self._user_weight = dict([(user_id, user_weight[idx])
            for user_id, idx in user_index.iteritems()])
        computed_hidden_feature = dict([(track_id, hidden_feature[idx])
            for track_id, idx in song_index.iteritems()])
        self._hidden_feature.update(computed_hidden_feature)
        self._res_norm = res_norm
        self._ratings = ratings
        self._user_models = {}
        # Make sure catalog includes tracks in `computed_hidden_feature`.
        for track_id in computed_hidden_feature:
            if not self.__catalog.get_track_by_id(track_id):
                self.__catalog[track_id] = Track(track_data={"id": track_id})

    def train_partial(self, ratings):
        """The incremental training of models.
        """
        raise NotImplementedError

    def get_user_model(self, user_id):
        return self._user_models.get(user_id)

    def train_user_taste_model(self, ratings):
        X = []
        y = []
        for track_id, rating in ratings.iteritems():
            track_hidden_features = self.get_track_hidden_features(track_id)
            if track_hidden_features is None:
                continue
            X.append(track_hidden_features)
            y.append(int(rating))
        user_model = train_user_taste_model(np.array(X), np.array(y))
        return user_model

    def update_user_model(self, user_id):
        ratings = self._ratings.get(user_id)
        assert ratings is not None, 'No ratings found for user %s' % user_id
        user_model = self.train_user_taste_model(ratings)
        self._user_models[user_id] = user_model
        return user_model

    def recommend_by_user_model(self, user_model, num=10):
        """Recommend n tracks from catalog based on user's taste model.

        :param catalog: a Catalog object.
        :param user_id: a string.
        :param num: the number of tracks to recommend.
        :return recommended_tracks: a list of Track objects.
        """
        pred_ratings = self.predict_all_ratings(user_model)
        sampled_track_ids = _sample_tracks_from_ratings(pred_ratings, num, None)
        return [self.__catalog[i] for i in sampled_track_ids]

    def recommend_by_tracks(self, seed_track_ids, num):
        raise NotImplementedError

    def recommend(self, user_id=None, seed_track_ids=None, num=10):
        if user_id is None and seed_track_ids is None:
            raise ValueError('Please specify either user_id or seed_track_ids!')
        if user_id is not None:
            user_model = self.get_user_model(user_id)
            if user_model is None:
                user_model = self.update_user_model(user_id)
            return self.recommend_by_user_model(user_model, num)
        else:
            return self.recommend_by_tracks(seed_track_ids, num)

    def get_track_hidden_features(self, track_id):
        assert hasattr(self, '_hidden_feature'), 'No hidden features found for track %s.' % track_id
        return self._hidden_feature.get(track_id)

    def get_user_ids(self, offset=0, limit=10):
        assert hasattr(self, '_ratings'), 'No ratings data provided.'
        return list(
            itertools.islice(self._ratings.iterkeys(), offset, offset+limit)
        )

    def get_track(self, track_id):
        return deepcopy(self.__catalog[track_id])

    def get_tracks(self, offset=0, limit=10):
        assert self.__catalog is not None
        track_ids = self.__catalog.get_track_ids(offset=offset, limit=limit)
        return [self.__catalog[i] for i in track_ids]

    def get_ratings_by_user(self, user_id):
        assert hasattr(self, '_ratings'), 'No ratings data provided.'
        return self._ratings.get(user_id)

    def predict_rating(self, user_taste_model, track):
        """
        :return rating: an integer from 1 to 5.
        """
        fea = self.get_track_hidden_features(track.id)
        return user_taste_model.predict(fea.reshape(1, -1))

    def predict_all_ratings(self, user_taste_model):
        """
        :return ratings: a dict mapping track_id to predicted rating.
        """
        ratings = dict(
            (track.id, self.predict_rating(user_taste_model, track))
            for track in self.__catalog.tracks()
        )
        return ratings

    def update_hidden_feature(self, input_hidden_feature):
        self._hidden_feature.update(input_hidden_feature)


def __rating_to_prob(rating):
    """Transform a rating of 1 to 5 to a non-negatie number proportational to
    its probability of being sampled.
    """
    # Exponential scale: one step higher in rating results in twice as much as
    # likely to be sampled.
    return float(2 ** rating)


def _sample_tracks_from_ratings(ratings, n, options):
    """
    :return track_ids: a list of string ids for tracks.
    """
    # TODO: allow filtering out certain track ids specified in `options`.
    track_ids_and_ratings = ratings.items()
    raw_rating_numbers = [x[1] for x in track_ids_and_ratings]
    probs = np.array(map(__rating_to_prob, raw_rating_numbers))
    probs = probs / max(probs.sum(), 1.)
    options = options or {}
    random_seed = options.get('random_seed')
    if random_seed is not None:
        np.random.seed(seed=random_seed)
    return np.random.choice([x[0] for x in track_ids_and_ratings],
                            size=n, p=probs, replace=False)


def recommend_by_user_model(user_model, hidden_feature, num_tracks):
    catalog = SimpleCatalog([{'id': track_id} for track_id in hidden_feature])
    engine = RecommendationEngine(catalog=catalog)
    engine.update_hidden_feature(hidden_feature)
    tracks = engine.recommend_by_user_model(user_model, num=num_tracks)
    return tracks or []


def train_user_model(ratings_one_user, hidden_feature):
    catalog = SimpleCatalog([{'id': track_id} for track_id in hidden_feature])
    engine = RecommendationEngine(catalog=catalog)
    engine.update_hidden_feature(hidden_feature)
    user_model = engine.train_user_taste_model(ratings_one_user)
    return user_model


def train_cf(all_ratings):
    engine = RecommendationEngine(catalog=None)
    engine.train(all_ratings)
    return engine.hidden_feature
