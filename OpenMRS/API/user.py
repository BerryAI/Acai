"""
    user.py
    ~~~
    This module provides all API for users

    :auther: Alexander Z Wang
"""

import sklearn
major_version, minor_version = tuple(sklearn.__version__.split('.')[:2])
if int(minor_version) >= 18 or int(major_version) >= 1:
    from sklearn.neural_network import MLPClassifier
    user_model_type = 'mlp'
else:
    from sklearn.linear_model import LogisticRegression
    user_model_type = 'logistic regression'


def train_user_taste_model(track_hidden_features, user_ratings):
    """Get user taste model for prediction

    :param track_hidden_feature: hidden feature matrix of listened tracks
    :param user_rating: user rating vector of listened tracks
    :return user_model: user taste model classifiers
    :rtype: classifiers
    """

    if user_model_type == 'logistic regression':
        user_model = LogisticRegression()
    elif user_model_type == 'MLP':
        user_model = MLPClassifier(
            hidden_layer_sizes=(5,), max_iter=3000, alpha=1e-2,
            algorithm='sgd', verbose=False, tol=1e-4, random_state=1,
            learning_rate_init=.01, activation='tanh')
    user_model.fit(X=track_hidden_features, y=user_ratings.ravel())
    return user_model


def predict_rating(user_model, track_hidden_features):
    """predict user rating score by user model

    :param user_model: user taste model classifiers
    :param track_hidden_features: hidden feature matrix of to be predicted
    :return prediction: predicted user rating score
    :rtype: list
    """
    prediction = user_model.predict(track_hidden_features.reshape(1, -1))
    prediction = prediction.tolist()
    return prediction
