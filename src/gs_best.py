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

from SGD_helpers import init_MF, matrix_factorization_SGD

best_gamma = 0.025
best_lambda_u = 0.1
best_lambda_i = 0.01
K = 20
num_epochs = 50

user_init, item_init = init_MF(ratings, K)
item_feats_SGD, user_feats_SGD, rmse = matrix_factorization_SGD(ratings, test, best_gamma, K, best_lambda_u, best_lambda_i, num_epochs,user_init, item_init)

tempt_dir = './'
np.save(tempt_dir + 'item_features_bias.npy',item_feats_SGD)
np.save(tempt_dir + 'user_features_bias.npy',user_feats_SGD)