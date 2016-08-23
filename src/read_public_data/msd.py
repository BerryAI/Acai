"""
    msd.py
    ~~~
    This module deal with Million Song Dataset

    :auther: Alexander Z Wang
"""

import io
import time


def read_tracks_database(filename):
    """Read all tracks information from Million Song Dataset

    :param filename: filename of the track information file.
    :return unique_tracks_info_dict: MSD tracks information.
    :rtype: dictionary
    """
    unique_tracks_info_dict = dict()
    unique_tracks_info_dict_reverse = dict()
    i = 0
    with io.open(filename, 'r', encoding='utf8') as fp:
        for line in fp:
            contents = line.rstrip('\n').split("<SEP>")
            track_info = contents[2] + "<SEP>" + contents[3]
            if track_info not in unique_tracks_info_dict:
                unique_tracks_info_dict[track_info] = i
                i = i+1
            if (i-1) not in unique_tracks_info_dict_reverse:
                unique_tracks_info_dict_reverse[i-1] = track_info

    return unique_tracks_info_dict, unique_tracks_info_dict_reverse


def get_song_ID_index(filename):
    """Read all tracks information from Million Song Dataset, return
    dictionaries of MSD ID with index and Song infromation with index

    :param filename: filename of the track information file.
    :return song_index: index of MSD ID of each song
    :return name_index: index of song infromation
    :rtype: dictionary
    """

    song_index = dict()
    name_index = dict()

    with io.open(filename, 'r') as fp:
        count = 0
        for line in fp:
            contents = line.rstrip('\n').split("<SEP>")
            track_ID = contents[1]
            if track_ID not in song_index:
                song_index[track_ID] = count
                if count not in name_index:
                    info = contents[2] + "<SEP>" + contents[3]
                    name_index[count] = info
                count += 1

    return song_index, name_index


def read_intersect_user_log(filename, unique_tracks_info_dict):
    """Read User Listening Logs intercept MSD

    :param filename: filename of the user listening log contains only MSD song
    :param unique_tracks_info_dict: MSD tracks information dictionary
    :return user_log_MSD: each user play history of MSD tracks.
    :return user_track_timestamp_MSD: timestamp information of each track
    :rtype: dictionary
    """
    user_log_MSD = dict()
    user_track_timestamp_MSD = dict()
    with io.open(filename, 'r', encoding='utf8') as fp:
        for line in fp:
            contents = line.rstrip('\n').rstrip('\r').split("\t")
            if len(contents) < 6:
                continue
            track_info = contents[3] + "<SEP>" + contents[5]
            if track_info not in unique_tracks_info_dict:
                continue
            if contents[0] in user_log_MSD:
                user_log_MSD[contents[0]].append(
                        unique_tracks_info_dict[track_info])
            else:
                user_log_MSD[contents[0]] = [
                        unique_tracks_info_dict[track_info]]
            if contents[0] in user_track_timestamp_MSD:
                if (unique_tracks_info_dict[track_info] in
                        user_track_timestamp_MSD[contents[0]]):
                    user_track_timestamp_MSD[contents[0]][
                        unique_tracks_info_dict[track_info]].append(
                            contents[1])
                else:
                    user_track_timestamp_MSD[contents[0]][
                        unique_tracks_info_dict[track_info]] = [contents[1]]
            else:
                track_timestamp_tmp = dict()
                track_timestamp_tmp[
                        unique_tracks_info_dict[track_info]] = [contents[1]]
                user_track_timestamp_MSD[contents[0]] = track_timestamp_tmp

    # Remove duplicated in user history
    for user in user_log_MSD:
        user_log_MSD[user] = list(set(user_log_MSD[user]))

    return user_log_MSD, user_track_timestamp_MSD


def get_track_rating_from_history(user_track_timestamp_MSD):
    """Calculating rates from users' listening history of MSD

    :param user_track_timestamp_MSD: timestamp information of each track
    :return user_rate_dict: each user's rating score
    :rtype: dictionary
    """
    time_format = "%Y-%m-%dT%H:%M:%SZ"
    user_rate_dict = dict()
    for user in user_track_timestamp_MSD:
        user_rate_dict[user] = dict()
        for key in user_track_timestamp_MSD[user]:
            length = len(user_track_timestamp_MSD[user][key])
            if length == 1:
                user_rate_dict[user][key] = 3
                continue

            # if a track played more than 10 times, 5 star rating
            if length > 10:
                user_rate_dict[user][key] = 5
                continue

            if length > 1:
                user_rate_dict[user][key] = 4

                # if a track played more than once in a single day, 5 star
                for i in range(0, length-1):
                    diff_time = abs(time.mktime(time.strptime(
                        user_track_timestamp_MSD[user][key][i], time_format)) -
                        time.mktime(time.strptime(
                            user_track_timestamp_MSD[user][key][i+1],
                            time_format))) / 3600
                    if diff_time < 24:
                        user_rate_dict[user][key] = 5
                        break
                if user_rate_dict[user][key] == 5:
                    continue

                # if a track played more than 4 times per month, 5 star rating
                if length > 4:
                    for i in range(0, length-4):
                        diff_time = abs(time.mktime(time.strptime(
                            user_track_timestamp_MSD[user][key][i],
                            time_format)) - time.mktime(time.strptime(
                                user_track_timestamp_MSD[user][key][i+3],
                                time_format))) / 3600 / 24
                        if diff_time < 30:
                            user_rate_dict[user][key] = 5
                            break
                if user_rate_dict[user][key] == 5:
                    continue

    return user_rate_dict
