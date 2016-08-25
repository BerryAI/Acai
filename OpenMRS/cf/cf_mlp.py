"""
    cf_mlp.py
    ~~~
    This module deploys MLP method from CF hidden features

    :auther: Alexander Z Wang
"""

from time import time
from sklearn.neural_network import MLPClassifier
import numpy


def get_user_training_data(
        hidden_feature, user_rate_dict, song_index, user_ID):
    """Get hidden feature sub matrix for each user

    :param hidden_feature: hidden feature matrix
    :param user_rate_dict: user rating matrix
    :param song_index: song index of hidden feature line number
    :param user_ID: user ID
    :return training_matrix: user submatrix of hidden features
    :return training_target: user rating history vector
    :rtype: ndarray
    """

    sub_matrix_tmp = []
    sub_rate_vector = []

    for key in user_rate_dict[user_ID]:
        line_number = song_index[key]
        sub_matrix_tmp.append(hidden_feature[line_number, :])
        sub_rate_vector.append(user_rate_dict[user_ID][key])

    training_matrix = numpy.array(sub_matrix_tmp)
    training_target = numpy.array(sub_rate_vector)

    return training_matrix, training_target


def get_user_prediction_MLP_hidden_feature(
        hidden_feature, user_rate_dict, song_index, user_ID):
    """Get hidden feature sub matrix for each user

    :param hidden_feature: hidden feature matrix
    :param user_rate_dict: user rating matrix
    :param song_index: song index of hidden feature line number
    :param user_ID: user ID
    :return user_prediction: prediction of user rating scores
    :rtype: ndarray
    """

    training_matrix, training_target = get_user_training_data(
            hidden_feature, user_rate_dict, song_index, user_ID)

    mlp = MLPClassifier(
        hidden_layer_sizes=(5,), max_iter=3000, alpha=1e-2,
        algorithm='sgd', verbose=False, tol=1e-4, random_state=1,
        learning_rate_init=.01, activation='tanh')
    mlp.fit(training_matrix, training_target)
    user_prediction = mlp.predict(hidden_feature)

    return user_prediction.tolist()


def get_predict_matrix_MLP_hidden_feature(
        hidden_feature, user_rate_dict, user_index, song_index):
    """Get predict matrix by MLP

    :param hidden_feature: hidden feature matrix
    :param user_rate_dict: user rating matrix (sparse)
    :param user_index: user ID index to row number of rating matrix
    :param song_index: song ID index to row number of hidden feature
    :return predict_matrix: prediction matrix
    :rtype: ndarray
    """

    predict_matrix = []

    inv_user_index = dict((v, k) for k, v in user_index.iteritems())

    start_time = time()
    for i in inv_user_index:
        user_ID = inv_user_index[i]
        user_prediction = get_user_prediction_MLP_hidden_feature(
                hidden_feature, user_rate_dict, song_index, user_ID)
        predict_matrix.append(user_prediction)
    end_time = time()
    print "MLP calculation time is: ", end_time-start_time

    return numpy.array(predict_matrix)


def get_user_profile(hidden_feature, user_rate_dict, song_index):
    """Get user profile of hidden feature weight by gradient descent method

    :param hidden_feature: hidden feature matrix
    :param user_rate_dict: user rating matrix
    :param song_index: index of song matrix
    :return user_profile: user weight profile
    :rtype: dictionary
    """

    user_profile = dict()

    for user_ID in user_rate_dict:
        track_hidden_features, user_ratings = get_user_training_data(
                hidden_feature, user_rate_dict, song_index, user_ID)
        user_model = train_user_taste_model(
            track_hidden_features, user_ratings)
        coef = (user_model.coefs_, user_model.intercepts_)
        user_profile[user_ID] = coef

    return user_profile
