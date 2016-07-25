"""
    cf_knn.py
    ~~~
    This module contains collaborative filtering algorithm based on KNN methods

    :auther: Alexander Z Wang
"""

import io
import operator


def get_knn_dict(full_user_his_dict, k):
    """Get user's other most k similar neighbours

    :param full_user_his_dict: dictionary of full user listening history
    :param k: number of similar users
    :return user_knn_dict: each user's k most similar neighbours
    :rtype: dictionary
    """
    user_knn_dict = dict()
    for user in full_user_his_dict:
        user_knn_dict[user] = []
        user_tmp_dict = dict()
        for another_user in full_user_his_dict:
            if user is another_user:
                continue
            user_tmp_dict[another_user] = len(
                set(full_user_his_dict[user]) &
                set(full_user_his_dict[another_user]))

        sorted_user_tmp_dict = sorted(
            user_tmp_dict.items(), key=operator.itemgetter(1))
        sorted_user_tmp_dict.reverse()
        boundary = 0
        for keys in sorted_user_tmp_dict:
            if boundary < k:
                user_knn_dict[user].append(keys)
                boundary = boundary + 1
            else:
                break
        print user_knn_dict[user]

    return user_knn_dict


def get_mean_vote_dict(user_rate_dict):
    """Calculating mean rates from users' listening history

    :param user_rate_dict: each user's rating score
    :return user_mean_votes_dict: each user's mean rating score
    :rtype: dictionary
    """

    user_mean_votes_dict = dict()
    for user in user_rate_dict:
        value = 0
        for key in user_rate_dict[user]:
            value += user_rate_dict[user][key]
        user_mean_votes_dict[user] = float(value) / float(
                                    len(user_rate_dict[user]))

    return user_mean_votes_dict


def write_neighbours(user_knn_dict, filename):
    """Write all neighbours information to disk

    :param user_knn_dict: dictionary of each user's k most similar neighbours
    :param filename: filename of neighbours' information
    """
    f = io.open(filename, "w")
    for user in user_knn_dict:
        tmp = user + "<SEP>"
        for other_user_value in user_knn_dict[user]:
            value = other_user_value[0] + ',' + str(other_user_value[1])
            tmp = tmp + value + "<SEP>"
        tmp.rstrip("<SEP>")
        tmp += "\n"
        f.write(tmp)

    f.close()


def get_knn_write_file(full_user_his, k, similar_weight_user_filename, max_k):
    """Get user's other most k similar neighbours and write to file

    :param full_user_his_dict: dictionary of full user listening history
    :param k: number of similar users
    :param similar_weight_user_filename: filename of neighbours information
    :param max_k: maximum number of neighbours could get, max_k > k
    :return user_knn_dict: each user's k most similar neighbours
    :rtype: dictionary
    """

    user_knn_dict = get_knn_dict(full_user_his, max_k)
    write_neighbours(user_knn_dict, similar_weight_user_filename)
    for user in user_knn_dict:
        user_knn_dict[user] = user_knn_dict[user][0:k]
    return user_knn_dict


def read_neighbours(filename, k):
    """Read user's most k similar neighbours from file

    :param filename: filename of neighbours' information
    :param k: number of similar users
    :return user_knn_dict: each user's k most similar neighbours
    :rtype: dictionary
    """

    user_knn_dict = dict()
    with io.open(filename, 'r') as fp:
        for line in fp:
            contents = line.rstrip("\n").split("<SEP>")
            tmp = []
            for i in range(0, k):
                values = contents[i+1].split(',')
                weight_and_neighbour = (values[0], float(values[1]))
                tmp.append(weight_and_neighbour)
            user_knn_dict[contents[0]] = tmp

    return user_knn_dict


def collaborative_filtering_user_based(
        user_knn_dict, user_log_MSD,
        user_rate_dict, user_mean_votes_dict, num):
    """Basic memory based collaboative filtering methods

    :param user_knn_dict: each user's k most similar neighbours
    :param user_log_MSD: each user's playing history of MSD tracks.
    :param user_rate_dict: each user's rating score
    :param user_mean_votes_dict: mean rating score of each user
    :param num: number of tracks to be recommended
    :return predict_dict: recommendation for each user based on MSD
    :rtype: dictionary
    """

    predict_dict = dict()
    for user in user_knn_dict:
        predict_dict[user] = []
        track_temp = []
        user_mean_tmp = user_mean_votes_dict[user]

        # combine all the tracks played by similar users
        for other_user in user_knn_dict[user]:
            track_temp += user_log_MSD[other_user[0]]
        track_temp = list(set(track_temp) - set(user_log_MSD[user]))

        # find values of each song and predict
        final_tmp = []
        for track in track_temp:
            value = 0
            weight = 0
            for other_user in user_knn_dict[user]:
                value += (
                    user_rate_dict[other_user[0]].get(
                        track, user_mean_votes_dict[other_user[0]]) -
                    user_mean_votes_dict[other_user[0]]
                            ) * float(other_user[1])
                weight += float(other_user[1])
            value = value / weight
            if value >= 4-user_mean_tmp:
                final_tmp.append(track)
        final_num = min(num, len(final_tmp))
        predict_dict[user] = sorted(final_tmp, reverse=True)[0:final_num]

    return predict_dict


def collaborative_filtering_knn_single_user(user, user_knn_dict,
                                            user_log_MSD, user_rate_dict,
                                            user_mean_votes_dict, num):
    """Basic memory based collaboative filtering methods

    :param user: the ID of user
    :param user_knn_dict: each user's k most similar neighbours
    :param user_log_MSD: each user's playing history of MSD tracks.
    :param user_rate_dict: each user's rating score
    :param user_mean_votes_dict: mean rating score of each user
    :param num: number of tracks to be recommended
    :return user_predict_list: recommendation for each user based on MSD
    :rtype: list
    """

    track_temp = []
    user_mean_tmp = user_mean_votes_dict[user]

    # combine all the tracks played by similar users
    for other_user in user_knn_dict[user]:
        track_temp += user_log_MSD[other_user[0]]
    track_temp = list(set(track_temp) - set(user_log_MSD[user]))

    # find values of each song and predict
    final_tmp = []
    for track in track_temp:
        value = 0
        weight = 0
        for other_user in user_knn_dict[user]:
            value += (user_rate_dict[other_user[0]].get(
                track, user_mean_votes_dict[other_user[0]]) -
                        user_mean_votes_dict[other_user[0]]
                        ) * float(other_user[1])
            weight += float(other_user[1])
        value = value / weight
        if value >= 4-user_mean_tmp:
            final_tmp.append((track, value))

    final_num = min(num, len(final_tmp))
    user_predict_list = sorted(
        final_tmp, key=operator.itemgetter(1), reverse=True)[0:final_num]

    return user_predict_list
