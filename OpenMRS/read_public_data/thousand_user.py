"""
    1k_user.py
    ~~~
    This module reads user playing history of 1k user

    :auther: Alexander Z Wang
"""
import io


def read_full_user_log(filename):
    """Read full listening history

    :param filename: filename of user listening logs.
    :return user_play_his_dict: each user's listening history
    :rtype: dictionary
    """
    song_dictionary = dict()
    user_play_his_dict = dict()
    count = 0
    with io.open(filename, 'r', encoding='utf8') as fp:
        for line in fp:
            contents = line.rstrip('\n').rstrip('\r').split("\t")
            if len(contents) < 6:
                continue
            track_info = contents[3] + "<SEP>" + contents[5]
            if track_info not in song_dictionary:
                song_dictionary[track_info] = count
                count = count + 1
            if contents[0] in user_play_his_dict:
                user_play_his_dict[contents[0]].append(
                    song_dictionary[track_info])
            else:
                user_play_his_dict[contents[0]] = [song_dictionary[track_info]]

    for user in user_play_his_dict:
        user_play_his_dict[user] = list(set(user_play_his_dict[user]))

    return user_play_his_dict
