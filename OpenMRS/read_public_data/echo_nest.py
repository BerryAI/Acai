"""
    echo_nest.py
    ~~~
    This module reads user playing history of Echo Nest

    :auther: Alexander Z Wang
"""

import io
import operator


def get_echo_nest_user_history(filename, song_index):
    """Read full listening history of Echo Nest Data

    :param filename: filename of echo nest data
    :param song_index: index of MSD ID of each song
    :return echo_nest_user_history: each user's listening history
    :rtype: dictionary
    """

    echo_nest_user_history = dict()
    with io.open(filename, 'r') as fp:
        for line in fp:
            contents = line.rstrip('\n').split("\t")
            if contents[1] not in song_index:
                continue
            if contents[0] not in echo_nest_user_history:
                echo_nest_user_history[contents[0]] = [(
                                    song_index[contents[1]], contents[2])]
            else:
                echo_nest_user_history[contents[0]].append((
                                    song_index[contents[1]], contents[2]))

    return echo_nest_user_history


def get_user_rating_from_history_echo_nest(echo_nest_user_history):

    """Calculating rates from users' listening history of echo nest

    :param echo_nest_user_history: each user's listening history
    :return user_rating_dict: each user's rating score
    :rtype: dictionary
    """

    user_rate_dict = dict()

    for user in echo_nest_user_history:
        user_rate_dict[user] = dict()
        for value in echo_nest_user_history[user]:
            score = 3
            play_times = int(value[1])
            if play_times > 4:
                score = 5
            if play_times < 5 and play_times > 1:
                score = 4
            index = value[0]
            if index not in user_rate_dict[user]:
                user_rate_dict[user][index] = score
            else:
                if user_rate_dict[user][index] < 5:
                    user_rate_dict[user][index] += 1

    return user_rate_dict


def get_top_user_rating_from_history_echo_nest(echo_nest_user_history, num):

    """Calculating rates from users' listening history of echo nest

    :param echo_nest_user_history: each user's listening history
    :param num: number of top user
    :return top_user_rating_dict: each user's rating score
    :rtype: dictionary
    """

    user_rating_dict = get_user_rating_from_history_echo_nest(
                                        echo_nest_user_history)

    user_rating_length = dict()
    for user in user_rating_dict:
        user_rating_length[user] = len(user_rating_dict[user])

    sorted_user_rating_length = sorted(
        user_rating_length.items(), key=operator.itemgetter(1), reverse=True)

    top_user_rating_dict = dict()
    for value in sorted_user_rating_length[0:num]:
        top_user_rating_dict[value[0]] = user_rating_dict[value[0]]

    return top_user_rating_dict
