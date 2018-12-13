import numpy as np
import scipy.sparse as sp

"""Check README for more details"""

def get_bias(ratings):
    """Compute bias matrix where 
    bias[:] = r_ui - mu - bias_u - bias_i"""
    nb_item, nb_user = ratings.shape
    # find the non-zero ratings indices 
    non_zero = ratings.nonzero()
    non_zero = list(zip(non_zero[0], non_zero[1]))
    
    # mu is overall mean
    mu = np.sum(ratings)/len(non_zero)

    ratings_by_user = np.zeros((nb_user,), dtype=int)
    ratings_by_item = np.zeros((nb_item,), dtype=int)
    
    # Compute number of non zero ratings by user and item
    for item, user in non_zero:
        ratings_by_user[user] = rate_user[user] + 1
        ratings_by_item[item] = rate_item[item] + 1
    
    # user_avg is mean by user
    # item_avg is mean by item
    user_avg = np.sum(ratings, axis = 0)
    item_avg = np.sum(ratings, axis = 1)
    user_avg = [user_avg[0, i] for i in range(nb_user)]/ratings_by_user
    item_avg = [item_avg[j, 0] for j in range(nb_item)]/ratings_by_item
    
    bias_users = user_avg - np.ones(nb_user) * mu
    bias_items = item_avg - np.ones(nb_item) * mu
    
    # compute the ratings matrix considering bias
    new_ratings = sp.lil_matrix((nb_item, nb_user))
    for item, user in non_zero:
        new_ratings[item, user] = ratings[item, user] - (mu + bias_users[user] + bias_items[item])
    return new_ratings, mu, bias_users, bias_items