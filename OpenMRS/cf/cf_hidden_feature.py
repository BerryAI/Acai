"""
    cf_hidden_feature.py
    ~~~
    This module contains collaborative filtering algorithm, based on matrix
    factorization to find hidden features in user rating. It includes SVD and
    Gradient Descent methods.

    :auther: Alexander Z Wang
"""

import numpy
import json


def full_rating_matrix_with_index(user_rate_dict):
    """Get full rating matrix with song index at each row

    :param user_rate_dict: user rate score dictionary (sparse)
    :return rating_matrix: full matrix of rating scores
    :rtype: dictionary
    """

    user_index = dict()
    song_index = dict()

    user_count = 0
    song_count = 0
    for user in user_rate_dict:
        if user not in user_index:
            user_index[user] = user_count
            user_count += 1
        for track_key in user_rate_dict[user]:
            if track_key not in song_index:
                song_index[track_key] = song_count
                song_count += 1

    rating_matrix = [None]*len(user_index)

    for user in user_rate_dict:
        rating_vector = [0.0] * len(song_index)
        for track_key in user_rate_dict[user]:
            rating_vector[song_index[track_key]] = user_rate_dict[user][
                track_key]
        rating_matrix[user_index[user]] = rating_vector

    rating_matrix = numpy.array(rating_matrix)
    matrix_update_by_song_mean_rate(rating_matrix)

    return user_index, song_index, rating_matrix


def matrix_update_by_song_mean_rate(rating_matrix):
    """Update rating score with average score

    :param rating_matrix: full matrix of rating scores
    """

    for i in range(0, len(rating_matrix[0])):
        index = rating_matrix[:, i] > 0
        ave_score = float(numpy.sum(rating_matrix[:, i])) / float(
            numpy.sum(index))
        for j in range(0, len(rating_matrix)):
            if rating_matrix[j][i] == 0.0:
                rating_matrix[j][i] = ave_score


def get_hidden_feature_matrix_SVD(user_rate_dict, k):
    """Get hidden feature matrix by SVD method

    :param user_rate_dict: each user's rating score
    :param k: number of hidden features
    :return data: hidden feature dataset
    :rtype: ndarray
    """

    user_index, song_index, rating_matrix = full_rating_matrix_with_index(
                                                user_rate_dict)

    U, s, V = numpy.linalg.svd(rating_matrix, full_matrices=True)

    V_bar = V[0:k]
    for i in range(0, k):
        V_bar[i] = s[i] * V_bar[i]
    hidden_feature = V_bar
    user_weight = U[:, 0:k]

    return user_weight, hidden_feature


def update_residue(rating_matrix, rate_bar):
    """update residue matrix for each iteration in Gradient descent method

    :param rating_matrix: users' rating matrix
    :param rate_bar: rating matrix generate by approximation in each GD step
    :return residue: residue matrix, rating_matrix - rate_bar
    :rtype: ndarray
    """

    residue = rating_matrix - rate_bar
    index = (rating_matrix == 0)
    residue[index] = 0

    return residue


def stochastic_GD(rating_matrix, lean_rate, lambda_rate, k, max_iter):
    """Stochastic Gradient Descent method

    :param rating_matrix: filename of unique MSD tracks
    :param lean_rate: learner rate
    :param lambda_rate: lambda rate
    :param k: number of hidden features
    :param max_iter: maximum iteration steps in gradient descent method
    :return user_weight: user weight matrix
    :return hidden_feature: hidden feature matrix
    :rtype: ndarray
    """

    m = len(rating_matrix)
    n = len(rating_matrix[0])

    user_weight = numpy.random.rand(m, k)
    hidden_feature = numpy.random.rand(n, k)

    rate_bar = user_weight.dot(hidden_feature.T)
    residue = update_residue(rating_matrix, rate_bar)

    res_norm = numpy.linalg.norm(residue)
    res_norm_list = [res_norm]

    for h in range(0, max_iter):

        user_weight = lean_rate*residue.dot(hidden_feature) + (
            1 - lean_rate*lambda_rate)*user_weight

        rate_bar = user_weight.dot(hidden_feature.T)
        residue = update_residue(rating_matrix, rate_bar)

        hidden_feature = lean_rate*residue.T.dot(user_weight) + (
            1 - lean_rate*lambda_rate)*hidden_feature

        rate_bar = user_weight.dot(hidden_feature.T)
        residue = update_residue(rating_matrix, rate_bar)

        res_norm = numpy.linalg.norm(residue)
        res_norm_list.append(res_norm)

        if res_norm < 0.01:
            break

    return user_weight, hidden_feature, res_norm_list


def stochastic_GD_with_ini(rating_matrix, user_weight, lean_rate,
                           hidden_feature, lambda_rate, max_iter):
    """Stochastic Gradient Descent method with given initail guess

    :param rating_matrix: filename of unique MSD tracks
    :param user_weight: user weight matrix
    :param hidden_feature: hidden feature matrix
    :param lean_rate: learner rate
    :param lambda_rate: lambda rate
    :param k: number of hidden features
    :return user_weight: user weight matrix
    :return hidden_feature: hidden feature matrix
    :return full_iteration: flag of iteration status
    :return res_norm_list: list of error norm of each iteration
    :rtype: ndarray
    """

    rate_bar = user_weight.dot(hidden_feature.T)
    residue = update_residue(rating_matrix, rate_bar)

    res_norm = numpy.linalg.norm(residue)
    res_norm_old = res_norm
    res_norm_list = []

    full_iteration = 1

    for h in range(0, max_iter):

        user_weight = lean_rate*residue.dot(hidden_feature) + (
            1 - lean_rate*lambda_rate)*user_weight

        rate_bar = user_weight.dot(hidden_feature.T)
        residue = update_residue(rating_matrix, rate_bar)

        hidden_feature = lean_rate*residue.T.dot(user_weight) + (
            1 - lean_rate*lambda_rate)*hidden_feature

        rate_bar = user_weight.dot(hidden_feature.T)
        residue = update_residue(rating_matrix, rate_bar)

        res_norm = numpy.linalg.norm(residue)
        res_norm_list.append(res_norm)

        if res_norm > res_norm_old:
            full_iteration = 0
            break
        if res_norm < 0.01:
            full_iteration = 2
            break
        res_norm_old = res_norm

    return user_weight, hidden_feature, res_norm_list, full_iteration


def stochastic_GD_r(rating_matrix, lean_rate, lambda_rate, k,
                    max_iter_inloop, max_iter_outloop):
    """Stochastic Gradient Descent method with flexible learner rate

    :param rating_matrix: filename of unique MSD tracks
    :param lean_rate: learner rate
    :param lambda_rate: lambda rate
    :param k: number of hidden features
    :return user_weight: user weight matrix
    :return hidden_feature: hidden feature matrix
    :rtype: ndarray
    """

    user_weight, hidden_feature, res_norm_list = stochastic_GD(
        rating_matrix, lean_rate, lambda_rate, k, max_iter_inloop)

    full_success = 1

    for i in range(0, max_iter_outloop):

        if full_success == 2:
            break
        if full_success == 1:
            lean_rate = 2*lean_rate
        if full_success == 0:
            lean_rate = lean_rate/2

        (user_weight, hidden_feature,
            res_norm_list_tmp, full_success) = stochastic_GD_with_ini(
                    rating_matrix, user_weight, lean_rate,
                    hidden_feature, lambda_rate, max_iter_outloop)

        res_norm_list = res_norm_list + res_norm_list_tmp

    return user_weight, hidden_feature, res_norm_list


def batch_GD(rating_matrix, lean_rate, lambda_rate, k, max_iter):
    """Batch Gradient Descent method

    :param rating_matrix: filename of unique MSD tracks
    :param lean_rate: learner rate
    :param lambda_rate: lambda rate
    :param k: number of hidden features
    :return user_weight: user weight matrix
    :return hidden_feature: hidden feature matrix
    :rtype: ndarray
    """

    residue = numpy.copy(rating_matrix)

    res_norm_old = numpy.linalg.norm(residue)
    res_norm_new = res_norm_old
    res_norm_list = [res_norm_old]

    m = len(rating_matrix)
    n = len(rating_matrix[0])

    user_weight = numpy.random.rand(m, k)
    hidden_feature = numpy.random.rand(n, k)

    columns = (residue != 0).sum(0)
    rows = (residue != 0).sum(1)
    diag_n = numpy.diag(1 - lean_rate*lambda_rate*columns)
    diag_m = numpy.diag(1 - lean_rate*lambda_rate*rows)

    for h in range(0, max_iter):
        user_weight = diag_m.dot(user_weight)
        user_weight += lean_rate * numpy.dot(residue, hidden_feature)
        hidden_feature = diag_n.dot(hidden_feature)
        hidden_feature += lean_rate * residue.T.dot(user_weight)
        rate_bar = user_weight.dot(hidden_feature.T)
        residue = update_residue(rating_matrix, rate_bar)
        res_norm_new = numpy.linalg.norm(residue)
        res_norm_list.append(res_norm_new)
        if res_norm_old < 1.0:
            break
        res_norm_old = res_norm_new

    return user_weight, hidden_feature


def get_hidden_feature_matrix_GD(
        user_rate_dict, k, lean_rate, lambda_rate, max_iter, GD_method):
    """Get hidden feature matrix by stochastic gradient descent method

    :param user_rate_dict: user rating matrix
    :param k: number of hidden features
    :param lean_rate: learner rate
    :param lambda_rate: lambda rate
    :param max_iter: max iteration steps in GD
    :param method: number of the method
    :return user_weight: user weight matrix
    :return hidden_feature: hidden feature matrix
    :rtype: ndarray
    """

    (user_index, song_index_hfmatrix,
        rating_matrix) = full_rating_matrix_with_index(user_rate_dict)

    if GD_method == 1:
        user_weight, hidden_feature, res_norm = stochastic_GD(
            rating_matrix, lean_rate, lambda_rate, k, max_iter)
    if GD_method == 2:
        user_weight, hidden_feature, res_norm = stochastic_GD_r(
            rating_matrix, lean_rate, lambda_rate, k, max_iter)
    if GD_method == 3:
        user_weight, hidden_feature, res_norm = batch_GD(
            rating_matrix, lean_rate, lambda_rate, k, max_iter)

    return (user_weight, hidden_feature, res_norm,
            user_index, song_index_hfmatrix)


def write_hidden_feature_to_file(hf_filename, hidden_feature, song_index):
    """Write hidden features to a Json file

    :param hf_filename: filename for hidden feature matrix
    :param hidden_feature: hidden feature matrix
    :param song_index: index of song in hidden feature matrix
    """

    inv_song_index = dict((v, k) for k, v in song_index.iteritems())
    data = dict()

    for key in inv_song_index:
        data[inv_song_index[key]] = hidden_feature[key].tolist()

    with open(hf_filename, 'w') as outfile:
        json.dump(data, outfile)


def get_user_profile(
        user_rate_dict, k, lean_rate, lambda_rate, max_iter, GD_method):
    """Get user profile of hidden feature weight by gradient descent method

    :param user_rate_dict: user rating matrix
    :param k: number of hidden features
    :param lean_rate: learner rate
    :param lambda_rate: lambda rate
    :param max_iter: max iteration steps in GD
    :param method: number of the method
    :return user_profile: user weight profile
    :rtype: dictionary
    """

    user_profile = dict()

    user_index, song_index, rating_matrix = full_rating_matrix_with_index(
                                            user_rate_dict)

    if GD_method == 1:
        user_weight, hidden_feature, res_norm = stochastic_GD(
            rating_matrix, lean_rate, lambda_rate, k, max_iter)
    if GD_method == 2:
        user_weight, hidden_feature, res_norm = stochastic_GD_r(
            rating_matrix, lean_rate, lambda_rate, k, max_iter)
    if GD_method == 3:
        user_weight, hidden_feature, res_norm = batch_GD(
            rating_matrix, lean_rate, lambda_rate, k, max_iter)

    for user in user_index:
        line_number = user_index[user]
        weight_tmp = user_weight[line_number, :].tolist()
        user_profile[user] = weight_tmp

    return user_profile
