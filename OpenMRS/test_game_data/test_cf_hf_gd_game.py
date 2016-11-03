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
import sys
import time
sys.path.append('../cf')
sys.path.append('../read_game_data')
import matplotlib.pyplot as plt
import acai_game as ag
import cf_hidden_feature as ch

k = 5
lean_rate = 0.001
lambda_rate = 0.04
max_iter = 10000
GD_method = 1

user_rate_dict = ag.get_user_rate_dict()

user_weight, hidden_feature, res_norm = ch.get_hidden_feature_matrix_GD(
                user_rate_dict, k, lean_rate, lambda_rate, max_iter, GD_method)
predict_matrix = user_weight.dot(hidden_feature.T)
print predict_matrix.shape
print res_norm[-1]

print "hidden features of 10 songs"
print hidden_feature[0:10, :]
hist, bin_edges = numpy.histogram(hidden_feature, bins=20)
print hist
print bin_edges

# Plot convergence
# plt.plot(res_norm)
# plt.ylabel('Norm of Error')
# plt.xlabel('Iteration Steps')
# plt.show()
