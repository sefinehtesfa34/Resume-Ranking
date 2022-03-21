import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.decomposition import NMF #Data colector
# Surprise 
import surprise
from surprise.reader import Reader
from surprise import Dataset
from surprise.model_selection import GridSearchCV
# CrossValidation
from surprise.model_selection import cross_validate
from surprise import SVD,NMF

np.random.seed(42) # replicating results
# Importing Online Data
# The work considers only tidy data in ratings.csv 
# and movies.csv. Specifically, ratings_df 
# records userId, movieId, and rating consecutively. 
# On the other hand, movies_df stores values in movieId and genres.

# movieId is, therefore, the mutual variable.

# Note that Surprise enables one to upload data, e.g. csv files,
#  for predictions through its own methods. On the other hand, 
#  as it is discussed below, Surprise also allows the user 
#  to use pandas' DataFrames. The author works with pd.DataFrame 
#  objects for convenience.

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

r = urlopen("http://files.grouplens.org/datasets/movielens/ml-latest-small.zip")
zipfile=ZipFile(BytesIO(r.read()))
# print the contentof zip file
zipfile.namelist()
# Tidy of ratings (movieId)
ratings_df=pd.read_csv(zipfile.open('ml-latest-small/ratings.csv'))
# print("Columns of ratongs_df: {0}".format(ratings_df.columns))
movies_df=pd.read_csv(zipfile.open('ml-latest-small/movies.csv'))
# print("Columns of movies_df: {0}".format(movies_df.columns))
# print(ratings_df.head())
# print(ratings_df.info())
# print(ratings_df.describe())
# print(movies_df.head())

#Note that movies_df contains only movieId and genres variables which store even multiple genres 
# separated by the vertical bar in one cell.
# Data preprocessing
# Filtering Data Set
# Firstly, it is essential to filter out movies and users with low exposure to 
# remove some of the noise from outliers. According to the official 
# MovieLens documentation, all selected users have rated at least 20 movies in the data set. 
# However, the following code filters out the movies and 
# users based on an arbitrary threshold 
# and creates a new data frame ratings_flrd_df. 
# Moreover, the chunk also prints the value of deleted movies 
# with new and old dimensions.

min_movie_ratings = 2 #a movie has was rated at least 
min_user_ratings =  5 #a user rated movies at least


ratings_flrd_df = ratings_df.groupby("movieId").filter(lambda x: x['movieId'].count() >= min_movie_ratings)
ratings_flrd_df = ratings_flrd_df.groupby("userId").filter(lambda x: x['userId'].count() >= min_user_ratings)
ratings_flrd_df.to_csv("ratings_flrd_df.csv",index=False)
print("{0} movies deleted; all movies are now rated at least: {1} times. Old dimensions: {2}; New dimensions: {3}"\
.format(len(ratings_df.movieId.value_counts()) - len(ratings_flrd_df.movieId.value_counts())\
        ,min_movie_ratings,ratings_df.shape, ratings_flrd_df.shape ))



reader=Reader(rating_scale=(0.5,5)) # line_format by default order of the fields
data=Dataset.load_from_df(ratings_flrd_df[["userId","movieId","rating"]],reader=reader)
trainset=data.build_full_trainset()
testset=trainset.build_anti_testset()

# Number of Factors and RMSE
# For the demonstrative purpose, let's examine the effect of number of latent factors k on the model\'s 
# performance. Specifically, it is possible to visually observe the effect of 
# multiple factors on error measurement. 
# As in supervised machine learning, 
# cross_validate computes the error rate for each fold. 
# The following function computes the average of RMSE given by the five folds and 
# append the empty list rmse_svd. Consequently, the list contains 100 measures of min RMSE 
# given 100 consecutive values of k in each test set, and by five folds in every iteration.


def rmse_vs_factors(algorithm, data):
  """Returns: rmse_algorithm i.e. a list of mean RMSE of CV = 5 in cross_validate() for each  factor k in range(1, 101, 1)
  100 values 
  Arg:  i.) algorithm = Matrix factoization algorithm, e.g SVD/NMF/PMF, ii.)  data = surprise.dataset.DatasetAutoFolds
  """
  
  rmse_algorithm = []
  
  for k in range(1, 101, 1):
    algo = algorithm(n_factors = k)
    
    #["test_rmse"] is a numpy array with min accuracy value for each testset
    loss_fce = cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=False)["test_rmse"].mean() 
    rmse_algorithm.append(loss_fce)
  
  return rmse_algorithm
rmse_svd = rmse_vs_factors(SVD,data)
# To replicate the plot of performance for each subsequent model, 
# the following chunk defines the function plot_rmse() with two arguments 
# where rmse is a list of float values and algorithm is an instantiated matrix factorization model. 
# The function returns a plot with two line subplots that display performance vs. numbers of factors. 
# The second subplot only zooms in and marks k with the best performance, i.e. the minimum RMSE.

def plot_rmse(rmse, algorithm):
  """Returns: sub plots (2x1) of rmse against number of factors. 
     Vertical line in the second subplot identifies the arg for minimum RMSE
    
     Arg: i.) rmse = list of mean RMSE returned by rmse_vs_factors(), ii.) algorithm = STRING! of algo 
  """
  
  plt.figure(num=None, figsize=(11, 5), dpi=80, facecolor='w', edgecolor='k')

  plt.subplot(2,1,1)
  plt.plot(rmse)
  plt.xlim(0,100)
  plt.title("{0} Performance: RMSE Against Number of Factors".format(algorithm), size = 20 )
  plt.ylabel("Mean RMSE (cv=5)")

  plt.subplot(2,1,2)
  plt.plot(rmse)
  plt.xlim(0,50)
  plt.xticks(np.arange(0, 52, step=2))

  plt.xlabel("{0}(n_factor = k)".format(algorithm))
  plt.ylabel("Mean RMSE (cv=5)")
  plt.axvline(np.argmin(rmse), color = "r")

# GridsearchCV (Sample)
param_grid = {'n_factors': [4,6,9,11,14,18,29]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5)
gs.fit(data)
# best RMSE score
print(gs.best_score['rmse'])
# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])


# output
# 0.8639552777419859
# {'n_factors': 11}
# To make the model generalizable, i.e. avoid over and underfitting, the grid algorithm finds n_factors = 11 optimal
# raining SVD Algorithm and Predictions
# Next, SVD(n_factors = 11) fits the model on trainset. 
# To predict values, i.e. ratings, for each empty element 
# aij
#  in the utility matrix, it is essential to specify: a.) 
#  the users and b.) particular movies that are not in the trainset.
#   build_anti_testset() method of trainset accomplishes the goal. 
#   It returns a list of ratings (testset) that are not in 
#   the trainset or in the entire utility matrix R
# . Consequently, it is possible to use the fitted model and predict ratings 
# for movies in testset. algo_SVD.test(testset)
# returns the list with predictions.

algo_SVD = SVD(n_factors = 11)
algo_SVD.fit(trainset)


# Predict ratings for all pairs (i,j) that are NOT in the training set.
testset = trainset.build_anti_testset()

predictions = algo_SVD.test(testset)

# subset of the list  predictions
predictions[0:2]


def get_top_n(predictions, userId, movies_df, ratings_df, n = 10):
    '''Return the top N (default) movieId for a user,.i.e. userID and history for comparisom
    Args:
    Returns: 
  
    '''
    #Peart I.: Surprise docomuntation
    
    #1. First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    #2. Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key = lambda x: x[1], reverse = True)
        top_n[uid] = user_ratings[: n ]
    
    #Part II.: inspired by: https://beckernick.github.io/matrix-factorization-recommender/
    
    #3. Tells how many movies the user has already rated
    user_data = ratings_df[ratings_df.userId == (userId)]
    print('User {0} has already rated {1} movies.'.format(userId, user_data.shape[0]))

    
    #4. Data Frame with predictions. 
    preds_df = pd.DataFrame([(id, pair[0],pair[1]) for id, row in top_n.items() for pair in row],
                        columns=["userId" ,"movieId","rat_pred"])
    
    
    #5. Return pred_usr, i.e. top N recommended movies with (merged) titles and genres. 
    pred_usr = preds_df[preds_df["userId"] == (userId)].merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId')
            
    #6. Return hist_usr, i.e. top N historically rated movies with (merged) titles and genres for holistic evaluation
    hist_usr = ratings_df[ratings_df.userId == (userId) ].sort_values("rating", ascending = False).merge\
    (movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId')
    
    
    return hist_usr, pred_usr

hist_SVD_124, pred_SVD_124 = get_top_n(predictions, movies_df = movies_df, userId = 124, ratings_df = ratings_df)

hist_SVD_124.head(15)





