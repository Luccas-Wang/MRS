#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

dat_dir = '../data/'
sub_dir = '../submit/'

from pre_post_process import load_data, split_data

_, ratings = load_data(dat_dir + "data_train.csv")
sample_ids, _ = load_data(dat_dir + "sample_submission.csv")
print(np.shape(ratings))

valid_ratings, train, test = split_data(ratings, p_test=0.1)

from MF_helpers import get_bias_train, get_bias_test

bias_train, total_bias, bias_u_train, bias_i_train = get_bias_train(train) #ratings for final submissions
bias_test = get_bias_test(test, total_bias, bias_u_train, bias_i_train)


from SGD_helpers import init_MF, matrix_factorization_SGD
# define parameters
gamma = 0.025
K = 20
lambda_user = 0.01
lambda_item = 0.1
num_epochs = 50

user_init, item_init = init_MF(bias_train, K)

item_featuresSGD, user_featuresSGD, rmse = matrix_factorization_SGD(bias_train, bias_test, gamma, K, lambda_user, lambda_item, num_epochs, user_init, item_init)

tempt_dir = './'
np.save(tempt_dir + 'item_features_bias.npy',item_featuresSGD)
np.save(tempt_dir + 'user_features_bias.npy',user_featuresSGD)