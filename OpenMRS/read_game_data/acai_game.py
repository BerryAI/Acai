"""
    acai_game.py
    ~~~
    This module get Acai game data to CF methods

    :auther: Alexander Z Wang
"""


import urllib
import urllib2
import json
import time


def get_track_info_by_trackID(track_ID):
    """Get artist name + track name from track_ID

    :param track_ID: Track ID
    :return track_info: artist name + track name
    :rtype: lsit of string
    """

    track_info = None
    url = "http://berry-music-cortex.appspot.com/api/tracks/" + track_ID
    response = urllib2.urlopen(url)
    data = json.load(response)
    if "error" not in data:
        track_info = data
    return track_info


def get_user_played_list_with_events(user_ID, **args):
    """Get user played list with events(like, disliked, etc)

    :param user_ID: user ID
    :param **args: arguments for calling api
        eventType: any values of [WTF, NotMyTaste, OK, Nice, LoveIt].
                   Leaving this value blank will get a history of played tracks
        page: Page number of the result. Default to 1.
        pageSize: Number of results to be returned from each page. Default 20
        timestamp_from: UTC timestamp of the oldest records to be returned.
                   Default to [1 hour ago]. This parameter should be always
                   provided to avoid time difference between host servers.
        timestamp_to: UTC timestamp of the newest record. Default to [now].
    :return user_play_list: played list with events
    :rtype: list
    """

    user_play_list = []
    query_param = dict()
    if args.has_key("eventType"):
        if args["eventType"] not in [
                "WTF", "NotMyTaste", "OK", "Nice", "LoveIt"]:
            return user_play_list
        query_param["eventType"] = args["eventType"]
    if args.has_key("page"):
        query_param["page"] = args["page"]
    if args.has_key("pageSize"):
        query_param["pageSize"] = args["pageSize"]
    if args.has_key("timestamp_from"):
        query_param["timestamp_from"] = args["timestamp_from"]
    if args.has_key("timestamp_to"):
        query_param["timestamp_to"] = args["timestamp_to"]
    query_param["user_id"] = user_ID

    base_url = "http://berry-acai.appspot.com/api/activities/?"
    url = base_url + urllib.urlencode(query_param)
    response = urllib.urlopen(url)
    data = json.loads(response.read())

    if "status" in data:
        if data["status"] == "failed":
            return user_play_list

    for value in data["track_ids"]:
        user_play_list.append(value["track_id"])

    return user_play_list


def get_user_rate_front_end(user_ID, timestamp_from=1464739200):
    """Get user rate for each track ever listened from front end

    :param user_ID: user ID
    :return user_rate: rating score of each track ever listened
    :rtype: dictionary
    """

    user_rate = dict()

    user_play_list = get_user_played_list_with_events(
        user_ID, timestamp_from=timestamp_from)
    if len(user_play_list) == 0:
        return user_rate
    for track in user_play_list:
        user_rate[track] = 3
    total_length = len(user_play_list)

    count = 0
    user_play_list_wtf = get_user_played_list_with_events(
        user_ID, eventType="WTF", timestamp_from=timestamp_from)
    for track in user_play_list_wtf:
        user_rate[track] = 1
    count += len(user_play_list_wtf)
    if count == total_length:
        return user_rate

    user_play_list_nmt = get_user_played_list_with_events(
        user_ID, eventType="NotMyTaste", timestamp_from=timestamp_from)
    for track in user_play_list_nmt:
        user_rate[track] = 2
    count += len(user_play_list_nmt)
    if count == total_length:
        return user_rate

    user_play_list_nice = get_user_played_list_with_events(
        user_ID, eventType="Nice", timestamp_from=timestamp_from)
    for track in user_play_list_nice:
        user_rate[track] = 4
    count += len(user_play_list_nice)
    if count == total_length:
        return user_rate

    user_play_list_love = get_user_played_list_with_events(
        user_ID, eventType="LoveIt", timestamp_from=timestamp_from)
    for track in user_play_list_love:
        user_rate[track] = 5

    return user_rate


def get_all_users():
    """Get all user from Acai game

    :return user_list: list of all users
    :rtype: list
    """

    target_url = 'http://acai.berry.ai/api/all_user_ids'
    data = urllib2.urlopen(target_url).read()
    user_list = data[1:-1].split(',')
    for i in range(0, len(user_list)):
        user_list[i] = user_list[i].strip(" ")[1:-1]

    return user_list


def get_user_rate_dict(timestamp_from=1464739200):
    """Get all user rate dictionary

    :return user_rate_dict: user rate score dictionary (sparse)
    :rtype: dictionary
    """

    user_rate_dict = dict()
    user_list = get_all_users()

    for user_ID in user_list:
        tmp_rate_dict = get_user_rate_front_end(
            user_ID, timestamp_from)
        if len(tmp_rate_dict) == 0:
            continue
        else:
            user_rate_dict[user_ID] = tmp_rate_dict
        time.sleep(0.2)

    return user_rate_dict
