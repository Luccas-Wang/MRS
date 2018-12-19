#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
from data_process import load_data, split_data
from SGD_helpers import init_MF, matrix_factorization_SGD

dat_dir = '../data/'
sub_dir = '../submit/'

from SGD_helpers import  init_MF, matrix_factorization_SGD


if __name__ == '__main__':
    
    
    # Load the data in sparse matrices
    print("Load user-item ratings...",flush=True,end='')
    ratings = load_data(dat_dir + "data_train.csv")
    print('end)
    
    print("Split data: 90% for training, 10% for testing...",flush=True,end='')
    valid_ratings, train, test = split_data(ratings, p_test=0.1)
    print('end)
         
    # Innitilize modle with the tuned hyperparameters by grid search
    print("Innitialize model...",flush=True,end='')
    best_gamma = 0.025
    best_lambda_u = 0.1
    best_lambda_i = 0.01
    K = 20
    num_epochs = 50

    user_init, item_init = init_MF(ratings, K)
    print('end)
          
    print("Start training...")
    item_feats_SGD, user_feats_SGD, rmse = matrix_factorization_SGD(ratings, test, best_gamma, K, best_lambda_u, best_lambda_i, num_epochs, user_init, item_init)
    
    
    # Compute predictions with best features matrices
    print("Start predicting...",flush=True,end='')
    from MF_helpers import predict_no_bias
    predictions =  predict_no_bias(item_feats_SGD, user_feats_SGD)
    print('end)
          
    # Set predictions above 5 (below 1) to 5 (1)
    predictions[ np.where(predictions > 5.0)] = 5.0
    predictions[ np.where(predictions < 1.0)] = 1.0
    
    # Convert matrix of predictions to array of the wanted ones
    print("Start submission...",flush=True,end='')
    from data_process import create_submission
    create_submission(predictions)
    print('end)
