"""
    test_cf_hf_gd.py
    ~~~
    This module contains testing function of SVD method to discover hidden
    features in collaborative filtering method

    In this tesing file, we generate rating matrix for 1k user playing history
    of songs in Million Song Dataset. Because there are large amount of miss
    match in two data source, we only generate rate matrix of tracks in MSD
    which are played by 1k user dataset. Then we user Gradient Descent method
    to discover the hidden features in the CF methods.

    :auther: Alexander Z Wang
"""
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append('./cf')
sys.path.append('./read_public_data')
import msd
import echo_nest as en
import cf_hidden_feature as ch

k = 5
lean_rate = 0.00001
lambda_rate = 0.00
max_iter = 5000
GD_method = 1

filename_tracks = "../../data/subset_unique_tracks.txt"
filename_echo_nest = "../../data/train_triplets.txt"
num = 1000

print "Reading MSD and Echo Nest Data..."
song_index, name_index = msd.get_song_ID_index(filename_tracks)
echo_nest_user_history = en.get_echo_nest_user_history(
                                filename_echo_nest, song_index)
user_rate_dict = en.get_top_user_rating_from_history_echo_nest(
                                echo_nest_user_history, num)

print "Calculating Hidden Features..."

user_weight, hidden_feature, res_norm = ch.get_hidden_feature_matrix_GD(
                user_rate_dict, k, lean_rate, lambda_rate, max_iter, GD_method)

print "hidden features of 10 songs"
print hidden_feature[0:10, :]
hist, bin_edges = numpy.histogram(hidden_feature, bins=20)
print hist
print bin_edges

# Plot convergence
plt.plot(res_norm)
plt.ylabel('Norm of Error')
plt.xlabel('Iteration Steps')
plt.show()
