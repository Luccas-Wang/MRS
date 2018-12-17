import surprise
import pandas as pd
import numpy as np
import re

dat_dir = '../data/'

file_path = dat_dir + 'data_train.csv'
ratings_df = pd.read_csv(file_path)
ratings_df.head()

r_c = np.array(list(map(lambda x:re.split("[r_c]", x), ratings_df.Id)))

ratings_df['User'] = r_c[:,1]
ratings_df['Item'] = r_c[:,3]

reader = surprise.Reader(rating_scale=(1, 5))
ratings = surprise.Dataset.load_from_df(ratings_df[['User', 'Item', 'Prediction']], reader)


from surprise import SVD
from surprise.model_selection import GridSearchCV

# 
data = ratings
# lr_all -> learning_rates or gamma, reg -> regularizer term or lambda 
param_grid = {'n_epochs': [30], 'n_factors':[20, 45, 100, 150], 'lr_all': [0.005],
              'reg_pu': [1.0, 0.1, 0.01, 0.001], 'reg_qi': [1.0, 0.1, 0.01, 0.001]} # Add 'biased': [False]

# ratings.split(n_folds=5): cv = 5
gs = GridSearchCV(SVD, param_grid, measures=['rmse', ], cv=5)
gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

# Best parameters for minimal RMSE: {'n_epochs': 30, 'n_factors': 150, 'lr_all': 0.005, 'reg_pu': 1.0, 'reg_qi': 0.001}

import pickle
with open('tuning_best_params','rb') as f:
    pickle.dump(gs.best_params['rmse'], f)
with open('tuning_gs','rb') as f:
    pickle.dump(gs, f)