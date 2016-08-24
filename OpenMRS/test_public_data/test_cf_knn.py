"""
    test_cf_knn.py
    ~~~
    This module contains testing function of CF-KNN method

    In this tesing file, we generate rating matrix for 1k user playing history
    of songs in Million Song Dataset. Because there are large amount of miss
    match in two data source, we then calculate k nearest neighbours purely
    from 1k user dataset. Then give the prediction for test user.

    :auther: Alexander Z Wang
"""
import os
import sys
sys.path.append('../cf')
sys.path.append('../read_public_data')
import cf_knn as ck
import thousand_user as tu
import msd

# filename of all track information of Million Song Dataset(MSD)
filename = "unique_tracks.txt"
# filename of all track information of subset MSD
filename_subset = "subset_unique_tracks.txt"
# filename of 1k user play history
user_log_filename = "userid-timestamp-artid-artname-traid-traname.tsv"
# filename of 1k user play history intersect MSD
user_log_intersection = "full_log.txt"
# filename to be written of similar users with weight
similar_weight_user_filename = "similar_user_weight.txt"
# randomly pick a test user from 1k user
test_user_name = "user_000691"
# number of songs to be recommend
recommended_num = 100
# number of similar neighbours
num_neighbours = 3
# number of maximum neighbours to be allowed
max_neighbours = 10

# get rating matrix part

# get MSD track dictionary
(unique_tracks_info_dict,
    unique_tracks_info_dict_reverse) = msd.read_tracks_database(
        filename_subset)
# get user play history within MSD
user_log_MSD, user_track_timestamp_MSD = msd.read_intersect_user_log(
    user_log_intersection, unique_tracks_info_dict)
# get user rating matrix
user_rate_dict = msd.get_track_rating_from_history(user_track_timestamp_MSD)
# get each user's mean rating score line
user_mean_votes_dict = ck.get_mean_vote_dict(user_rate_dict)

# get nearest neighbours
if os.path.isfile(similar_weight_user_filename):
    user_knn_dict = ck.read_neighbours(
        similar_weight_user_filename, num_neighbours)
else:
    full_user_his = tu.read_full_user_log(user_log_filename)
    user_knn_dict = ck.get_write_knn(
        full_user_his, num_neighbours,
        similar_weight_user_filename, max_neighbours)
    ck.write_neighbours(user_knn_dict, similar_weight_user_filename)

# find recommendation for test user
user_predict_list = ck.collaborative_filtering_knn_single_user(
    test_user_name, user_knn_dict, user_log_MSD,
    user_rate_dict, user_mean_votes_dict, recommended_num)

print "The recommendation list for ", test_user_name, "is "
print user_predict_list
