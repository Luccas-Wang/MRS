#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

dat_dir = '../data/'
sub_dir = '../submit/'

from pre_post_process import *

_, ratings = load_data(dat_dir + "data_train.csv")
sample_ids, _ = load_data(dat_dir + "sample_submission.csv")
print(np.shape(ratings))

valid_ratings, train, test = split_data(ratings, p_test=0.1)

from SGD_helpers import *

# Finding gamma:
gamma = 0.025
Ks = np.linspace(10,100,num=10)
lambda_user = 0.1
lambda_item = 0.01
num_epochs = 50
errors = []

for index,  K in enumerate(Ks):
    # Initialize features matrix
    user_init, item_init = init_MF(train, K)
    # Compute SGD
    item_feature, user_feature, rmse = matrix_factorization_SGD(train, test, gamma,
                                          K, lambda_user, lambda_item,
                                          num_epochs, user_init, item_init)
    predictions = np.dot(item_feats_SGD.T, user_feats_SGD)
    predictions[ np.where( predictions > 5.0 ) ] = 5.0
    predictions[ np.where(predictions < 1.0)] = 1.0
    wanted_preds = getWantedPredictions(predictions.T, sample_ids)
    create_csv_submission(sample_ids, np.round(wanted_preds), sub_dir + 'res_'+str(index)+'.csv')

    errors.append(rmse)
    np.save('item_feature_'+ str(index) +'.npy', item_feature)
    np.save('user_feature_'+ str(index) +'.npy', user_feature)
    np.save('rmse_gamma.npy', errors)