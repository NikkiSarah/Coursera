## Limitations of sklearn's non-negative matrix factorisation library
# 1. Load the movie ratings data (as in the HW3-recommender-system) and use matrix factorisation technique(s) and
# predict the missing ratings from the test data. Measure the RMSE.
# 2. Discuss the results and why sklearn's non-negative matrix factorisation library did not work well compared to
# simple baseline or similarity-based methods we’ve done in Module 3. Can you suggest a way(s) to fix it?

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.sparse import csr_matrix, csr_array
from sklearn.decomposition import MiniBatchNMF, NMF
from sklearn.metrics import mean_squared_error

# load the data
MV_users = pd.read_csv('data/movie-lens-1m/users.csv')
MV_movies = pd.read_csv('data/movie-lens-1m/movies.csv')
train = pd.read_csv('data/movie-lens-1m/train.csv')
test = pd.read_csv('data/movie-lens-1m/test.csv')

MV_users.rename(columns={'accupation': 'occupation'}, inplace=True)

# unlike before, let's do a little EDA on the data
# most movies have been rated a 3, 4 or 5, with 4 the most common
sns.countplot(train, x="rating", color="seagreen")
plt.title("Distribution of rating counts")
sns.despine()

# it's probably unsurprising to find that there is a positive relationship between the count and release year
sns.countplot(MV_movies, y="year", color="seagreen")
plt.title("Distribution of movie counts by release year")
sns.despine()

# after some manipulation, we can see that "Drama" and "Comedy" are the most common genre tags (a movie can have
# multiple genres) and "Film-Noir" and "Western" are the least-common tags
MV_movies_long = pd.melt(MV_movies, id_vars=['mID', 'title', 'year'], var_name='genre', value_name='count')
MV_movies_long = MV_movies_long.loc[MV_movies_long['count'] == 1, ]
MV_movies_long.drop('count', axis=1, inplace=True)

genre_dict = {'Doc': 'Documentary',
              'Com': 'Comedy',
              'Hor': 'Horror',
              'Adv': 'Adventure',
              'Wes': 'Western',
              'Dra': 'Drama',
              'Ani': 'Animation',
              'War': 'War',
              'Chi': "Children's",
              'Cri': 'Crime',
              'Thr': 'Thriller',
              'Sci': 'Science-Fiction',
              'Mys': 'Mystery',
              'Rom': 'Romance',
              'Fil': 'Film-Noir',
              'Fan': 'Fantasy',
              'Act': 'Action',
              'Mus': 'Musical'}
MV_movies_long['genre'] = MV_movies_long.genre.map(genre_dict)

sns.countplot(MV_movies_long, y="genre")
plt.title("Distribution of movie counts by genre")
sns.despine()

# the number of times each genre was rated is very similar to the overall genre counts
MV_movies_ratings = MV_movies_long.merge(train, on='mID')

sns.countplot(MV_movies_ratings, y="genre")
plt.title("Distribution of movie rating counts by genre")
sns.despine()

# ... but interestingly doesn't necessarily correlate with the average rating - in fact Documentaries and Film-Noir
# had the highest average rating and Horror and Children's the lowest
average_ratings = MV_movies_ratings.groupby('genre')['rating'].mean().reset_index()
average_ratings.sort_values(by='rating', inplace=True)

sns.barplot(average_ratings, x='rating', y='genre', color='seagreen')

# the most common occupations in the dataset are Executive/Managerial, Other and College/Grad student. The least
# common occupation is Farmer
occupation_dict = {4: 'College/Grad student',
                   0: 'Other/Not specified',
                   7: 'Executive/Managerial',
                   1: 'Academic/Educator',
                   17: 'Technician/Engineer',
                   12: 'Programmer',
                   14: 'Sales/Marketing',
                   20: 'Writer',
                   2: 'Artist',
                   16: 'Self-employed',
                   6: 'Doctor/Health-care',
                   10: 'K-12 student',
                   3: 'Clerical/Admin',
                   15: 'Scientist',
                   13: 'Retired',
                   11: 'Lawyer',
                   5: 'Customer service',
                   9: 'Homemaker',
                   19: 'Unemployed',
                   18: 'Tradesman/Craftsman',
                   8: 'Farmer'}
MV_users['occupation'] = MV_users.occupation.map(occupation_dict)

sns.countplot(MV_users, y="occupation")
plt.title("Distribution of user counts by occupation")
sns.despine()

# the median age in the dataset are 25-34 year olds
sns.countplot(MV_users, y="age", color="seagreen")
plt.title("Distribution of user counts by age")
sns.despine()

# set up objects
allusers = list(MV_users.uID)
allmovies = list(MV_movies.mID)
mid2idx = dict(zip(MV_movies.mID, list(range(len(MV_movies)))))
uid2idx = dict(zip(MV_users.uID, list(range(len(MV_users)))))
FILL_VALUE = 3.5

# sample the data
recsys_train = train[:30000]
recsys_test = test[:30000]
# use all the data
recsys_train = train.copy()
recsys_test = test.copy()

def set_up_recsys(recsys_train, recsys_test):
    # identify all the users and movies in the sampled training and testing data combined
    recsys_MV_users = MV_users[(MV_users.uID.isin(recsys_train.uID)) | (MV_users.uID.isin(recsys_test.uID))]
    recsys_MV_movies = MV_movies[(MV_movies.mID.isin(recsys_train.mID)) | (MV_movies.mID.isin(recsys_test.mID))]

    # explode into sparse matrices with user ids as the rows and movie ids as the columns
    ind_movie = [mid2idx[id] for id in recsys_train.mID]
    ind_user = [uid2idx[id] for id in recsys_train.uID]
    rating_train = list(recsys_train.rating)
    Mr = np.full(shape=(len(allusers), len(allmovies)), fill_value=FILL_VALUE)
    Mr[ind_user, ind_movie] = rating_train

ind_movie = [mid2idx[id] for id in recsys_test.mID]
ind_user = [uid2idx[id] for id in recsys_test.uID]
rating_test = list(recsys_test.rating)
yt = np.full(shape=(len(allusers), len(allmovies)), fill_value=FILL_VALUE)
yt[ind_user, ind_movie] = rating_test
yt = csr_array(yt)

#double check the matrices are correct (if the FILL_VALUE != 0 then the assertions fail)
Mr_check = csr_matrix(recsys_train.pivot(index='uID', columns='mID', values='rating').fillna(FILL_VALUE))
yt_check = csr_matrix(recsys_test.pivot(index='uID', columns='mID', values='rating').fillna(FILL_VALUE))

assert Mr.sum() == Mr_check.sum()
assert yt.sum() == yt_check.sum()

# NMF was developed from PCA and is designed to extract sparse and significant features from a set of non-negative
# data factors. It decomposes a non-negative matrix (Mr) into its user (W) and movie (H) component matrices by
# optimising the distance between Mr and the matrix product of the component matrices. The most widely used distance
# measure (and the default in sklearn) is the squared Frobenius norm.
# The number of latent or hidden features must be pre-specified, and is the second dimension of the component matrices.
# It can be at most equal to the total number of movies in Mr.
init_method = [None, 'random', 'nndsvd', 'nnsvda']
BATCH_SIZE = 1024 # larger batches give better long-term convergence at the cost of a slower start
MAX_ITER = 500

for val in [5, 10, 20, 50]:
    # model = NMF(n_components=val, init=init_method[2], random_state=42, max_iter=MAX_ITER)
    model = MiniBatchNMF(n_components=val, init=init_method[2], random_state=42, max_iter=MAX_ITER,
                         batch_size=BATCH_SIZE)

    W = model.fit_transform(Mr)     # shape (#users, #latent features)
    H = model.components_           # shape (#latent features, #movies)

    # construct the prediction matrix
    yp = np.dot(W, H)
    # apply a threshold to coerce small values to be 0
    # yp[yp < 1] = 0
    yp = csr_array(yp)

    # identify non-missing actual ratings in the dataset
    # non_missing_idx = yt.nonzero()
    non_missing_idx = (yt != FILL_VALUE).nonzero()
    rmse = np.sqrt(mean_squared_error(yt[non_missing_idx], yp[non_missing_idx]))

    print('Latent features:', val, '; RMSE:', round(rmse, 6))

# Recall that the results from the previous recommender systems were:
# |Method|RMSE sample|RMSE|
# |:----|:--------:|:--:|
# |Baseline, $Y_p=3$|1.2643|1.2586|
# |Baseline, $Y_p=\mu_u$|1.1430|1.0353|
# |Content based, item-item|1.1963|1.0128|
# |Collaborative, cosine|1.1430|1.0263|
# |Collaborative, jaccard, $M_r\geq 3$| |0.9819|
# |Collaborative, jaccard, $M_r\geq 1$| |0.9914|
# |Collaborative, jaccard, $M_r$| |0.9517|

# |Method|RMSE sample|RMSE|
# |:----|:--------:|:--:|
# |Collaborative, 0 fill|3.73| |
# |Collaborative, 3.5 fill|1.12| |

# The RMSE on the sample data was 3.7 and the RMSE on the full dataset was x. NMF performs poorly because 'missing'
# ratings are automatically set to 0, which is outside the range of valid values (1-5) and adversely affects the loss
# Frobenius norm function. Changing the fill value to a non-zero value appeared to help immensely, but a single value
# had to be used, which isn't ideal as it's a relatively crude approach. The best solution would be to simply avoid
# including missing rating data in the loss function calculation, but that's not possible using the sklearn
# implementation of NMF.

# What could also help would be increasing the number of components (although at lower values, RMSE appeared to be
# relatively insensitive to changes). This wasn't experimented with here due to computational resource (power and time)
# constraints.
