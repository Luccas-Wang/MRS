#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

dat_dir = '../data/'
sub_dir = '../submit/'

from pre_post_process import load_data, split_data

_, ratings = load_data(dat_dir + "data_train.csv")
sample_ids, _ = load_data(dat_dir + "sample_submission.csv")
print(np.shape(ratings))

valid_ratings, train, test = split_data(ratings, p_test=0.)

from bias_helpers import computeBiasMatrix
bias_train, mean, bias_u_train, bias_i_train = computeBiasMatrix(train) #ratings for final submissions
bias_test, _, _, _ = computeBiasMatrix(test)

from SGD_helpers import init_MF, matrix_factorization_SGD
# define parameters
gamma = 0.025
K = 20
lambda_user = 0.01
lambda_item = 0.1
num_epochs = 50
user_init, item_init = init_MF(bias_train, K)

item_featuresSGD, user_featuresSGD, rmse = matrix_factorization_SGD(bias_train, bias_test, gamma, K, lambda_user, lambda_item, num_epochs, user_init, item_init)

import pickle
tempt_dir = './'
with open(tempt_dir + 'item_features_bias.pk','wb') as f:
    pickle.dump(item_featuresSGD, f)
with open(tempt_dir + 'user_features_bias.pk','wb') as f:
    pickle.dump(user_featuresSGD, f)