# EPFL CS-433 - Project II: Recommender System

## Contents
- README.MD (this)
- submit
  - submission.csv (with RMSE of 1.025)
- data
  - data_train.csv (original data)
  - sample_submission.csv (original data)
  - item_feats_SGD.npy (latent features of items after training using SGD)
  - user_feats_SGD.npy (latent features of users after training using SGD)
- src
  - data_process.py (module for data preprocess and results submission)
  - SGD_helpers.py (module for SGD, mostly matching the lab)
  - MF_helpers.py (module for the bias matrix factorization and user-item ratings matrix)
  ---
  - run.py (main script for submiting the final results)
  - run.py (main script for submiting the final results)

## Codes
### Prerequisites
- Python 3.6+
- Numpy
- Scipy
- Pandas

> #### Note for running implement_surprise.py, some other libraries are required. Please refer to the script for details.

- 1 iPython Notebooks: Recommender_Collaborative.ipynb, Recommender_Factorization.ipynb, Recommender_Surprise.ipynb
- 5 python modules : pre_post_process.py, collaborative.py, SGD_helpers.py, bias_helpers.py run.py

_Note_ : in order to do the preprocessing you will need the files "data_train.csv" and "sample_submission.csv" of the Kaggle competition available at this link: https://www.kaggle.com/c/epfml17-rec-sys

Used Libraries: 
- Surprise: http://surpriselib.com/ All informations about installations are provided at this link (pip install scikit-surprise) and the documentation is really detailed. Our Recommender_Surprise.ipynb notebook, described below, shows a nice and not too complex use of the library.
- Pandas: https://pandas.pydata.org/ (pip install pandas or conda install pandas with Anaconda). We only used pandas to load the ratings in a dataframe to be able to use the Surprise.Dataset.load_from_df function that needed a pandas.DataFrame as input.
- Sklearn: http://scikit-learn.org/stable/install.html for the installation. We used this library only to compute the similarities matrices between users in the Recommender_Collaborative.ipynb notebook.
- Numpy / Scipy: https://www.scipy.org/install.html for the installation (already available with Anaconda).
- Seaborn: https://seaborn.pydata.org/installing.html#installing for the installation. Seaborn was only used to visualize the grid search (for the report).
				
1. Report : this is where you will discover how we tackled the competition.

2. Notebooks : 

	- xxx.ipynb: Notebook using the User - user collaborative filtering technique. Here are the following steps of the notebook:
		
		1. Load the data, divide the ratings matrix in training and testing set.
		2. Statistics on the dataset
		3. Presentation of the algorithm used: similarity metrics, how to compute the predictions
		4. Find best parameters (similarity matrix and number K of best neighbors to keep for each user) by running the algorithm numerous times with different parameters: train on training ratings and test on testing ratings with RMSE.
		5. Compute the wanted predictions on the whole ratings (no test), using the algorithm with the best found parameters.
		6. Creation of csv file for the submission.

3. Python modules :

    - pre_post_process.py : This module allows us to transform the data of a csv file into a sparse ratings matrix and functions to convert the final made predictions in a csv file of the correct format expected by the Kaggle competition. It contains the following important functions:
							
			- preprocess_data(data):
				Parameters:
				Returns:
				
				
    - run.py : This module executes the program
    
To run the run.py module :
- On Mac : open the Terminal, enter your path to the folder where the Python modules are, enter the following command : chmod +x run.py. To execute enter : python run.py
- On Windows: open the Terminal, enter your path to the folder where the Python modules are. To execute enter : python run.py
