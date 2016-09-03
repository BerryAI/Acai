"""
    test_cf_hd_svd.py
    ~~~
    This module contains testing function of SVD method to discover hidden
    features in collaborative filtering method

    In this tesing file, we generate rating matrix for 1k user playing history
    of songs in Million Song Dataset. Because there are large amount of miss
    match in two data source, we only generate rate matrix of tracks in MSD
    which are played by 1k user dataset. Then we user SVD method to discover
    the hidden features in the CF methods.

    :auther: Alexander Z Wang
"""

import numpy
import cf_hidden_feature as ch
import msd

# filename of all track information of subset of Million Song Dataset(MSD)
filename_subset = "subset_unique_tracks.txt"
# filename of 1k user play history intersect MSD
user_log_intersection_filename = "full_log.txt"
# number of hidden features
k = 5

# get MSD track dictionary
(unique_tracks_info_dict,
    unique_tracks_info_dict_reverse) = msd.read_tracks_database(filename_subset)
# get user play history within MSD
user_log_MSD, user_track_timestamp_MSD = msd.read_intersect_user_log(
    user_log_intersection_filename, unique_tracks_info_dict)
# get user rating dictionary
user_rate_dict = msd.get_track_rating_from_history(user_track_timestamp_MSD)

# get hidden feature matrix
user_weight, hidden_feature = ch.get_hidden_feature_matrix_SVD(user_rate_dict
                                    , k)
print "hidden features of first 10 tracks"
print hidden_feature[0:10,:]

hist, bin_edges = numpy.histogram(hidden_feature, bins=20)
print hist
print bin_edges
