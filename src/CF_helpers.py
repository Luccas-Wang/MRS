# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

### Check the README.md for more details, especially on parameters

def get_users_per_item(sample_ids):
    """Convert sample_ids into np.array 
    containing the lists of users associated to the corresponding item"""
    result = []
    user_list = []
    current_item = 0
    for user, item in (sample_ids - 1):
        if item == current_item:
            user_list.append(user)
        else:
            result.append(user_list)
            user_list = [user]
            current_item = item
    result.append(user_list)
    return np.array(result)

def get_K_best(similarity, K):
    """Given the similarities of a user as an array,
    keep the K best similarities,
    set other similarities to 0"""
    similarity_copy = np.copy(similarity)
    index = len(similarity) - K
    # Get the similarity value of the K best neighbor
    best_neighbor = np.partition(similarity, index)[index]
    similarity_copy[similarity_copy < best_neighbor] = 0.0
    return similarity_copy

def prediction_by_item(ratings, item, similarity, user, K_best):
    """return in an array the predictions for the given movie and return:
    Parameters: 
    ratings:  The ratings matrix with dimension: item x user
    item: Index of item we want to predict (from 0 to 9999)
    similarity: The similarity matrix between all users
    user: The list of users to compute the predictions
    K_best: The number of best neighbors kept """
    ratings = ratings[item]
    # Following steps: only keep the similarities (user) of the given users
    Ratings = np.tile(ratings, (len(user), 1))
    Ratings[Ratings != 0] = 1
    # Get similarity matrix of the wanted users with dimension: all users (10000) x wanted users(< 10000)
    similarity_wanted = np.multiply(similarity, similarity[user])
    
    # Apply function getTopKNeighbors to each column of similarities to keep K best similarities for each user
    best_neighbor = np.apply_along_axis(get_K_best, 0, similarity_wanted, K_best)
    # Get a list of sums of each column of bestNeighbors
    sum_similarity = np.sum(best_neighbor, axis = 0)

    #Compute prediction: dot product of the items in ratings @ bestNeighbors divided by the corresponding sum of similarities
    predict = np.dot(ratings, best_neighbor)
    predict = predict / sum_similarity
    predict[np.isnan(predict)] = 0.0
    return predict

def compute_matrix_predictions(ratings, ratings_to_predict, similarity_matrix, K_best):
    """Compute the wanted predictions by running the compute_predictions_movie for each movie"""
    nb_users, nb_movies = ratings.shape
    movie_user_matrix = np.copy(ratings).T 
    predictions = []
    ### Compute wanted predictions for wanted users
    for movie in range(nb_movies):
        users_to_predict = ratings_to_predict[movie]
        if(len(users_to_predict) > 0):
            predictions_movie = compute_predictions_movie(movie_user_matrix, movie, similarity_matrix, users_to_predict, K_best)
            predictions.append(predictions_movie)
    return predictions