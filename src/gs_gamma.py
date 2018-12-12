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
gammas = np.logspace(-5,0,6)
K = 50
lambda_user = 0.01
lambda_item = 0.01
num_epochs = 20
errors = []

for gamma in gammas:
    # Initialize features matrix
    user_init, item_init = init_MF(train, K)
    # Compute SGD
    _, _, rmse = matrix_factorization_SGD(train, test, gamma,
                                          K, lambda_user, lambda_item,
                                          num_epochs, user_init, item_init)
    errors.append(rmse)
    
np.save('rmse_gamma.npy', errors)