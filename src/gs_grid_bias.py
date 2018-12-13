#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

dat_dir = '../data/'
sub_dir = '../submit/'

from pre_post_process import load_data, split_data

_, ratings = load_data(dat_dir + "data_train.csv")
sample_ids, _ = load_data(dat_dir + "sample_submission.csv")
print(np.shape(ratings))

_, train, test = split_data(ratings, p_test=0.1)

from SGD_helpers import init_MF, matrix_factorization_SGD

from MF_helpers import get_bias_train, get_bias_test

bias_train, overal_bias, bias_u_train, bias_i_train = get_bias_train(train) #ratings for final submissions
bias_test = get_bias_test(test, overal_bias, bias_u_train, bias_i_train)

# Grid Search:
grid = np.zeros((3, 4, 4))
gamma = 0.025 # best gamma we found above
num_epochs = 20
lambdas_user = np.logspace(-3,0,4)[::-1] #From max to min
lambdas_item = np.logspace(-3,0,4)[::-1]
num_features = np.array([20, 50, 100])
min_loss = 100000
best_user_features = []
best_item_features = []

for x,K in enumerate(num_features):
    ### Warm start: directly start computation from previously computed item_features and user_features and not random initialization 
    user_init, item_init = init_MF(train, int(K))
    for y,lambda_u in enumerate(lambdas_user):
        for z,lambda_i in enumerate(lambdas_item):
            print("K = {}, lambda_u = {}, lambda_i = {}".format(int(K), lambda_u, lambda_i))
            item_feats, user_feats, rmse = matrix_factorization_SGD(bias_train, bias_test, gamma, int(K), lambda_u, lambda_i, num_epochs, user_init, item_init)
            ### For warm start, we keep the user_features and item_features that gave us the minimal rmse previously computed
            if rmse < min_loss:
                min_loss = rmse
#                 user_init = user_feats
#                 item_init = item_feats
                best_user_features = np.copy(user_feats)
                best_item_features = np.copy(item_feats)
            grid[x, y, z] = rmse
        np.save('grid.npy', grid)
        np.save('best_user_features_bias.npy', best_user_features)
        np.save('best_item_features_bias.npy', best_item_features)