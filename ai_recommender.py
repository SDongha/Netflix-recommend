

import pandas as pd
import numpy as np
import os
from scipy.sparse.linalg import svds

data_path = '/content/drive/MyDrive/AI_P2'
movies_filename = '/content/drive/MyDrive/AI_P2/movie.csv'
ratings_filename = '/content/drive/MyDrive/AI_P2/rating.csv'

df_movies = pd.read_csv(
    os.path.join(data_path, movies_filename),
    usecols=['movieId', 'title'],
    dtype={'movieId': 'int32', 'title': 'str'})

df_ratings = pd.read_csv(
    os.path.join(data_path, ratings_filename),
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

df_ratings.shape

df_ratings.head()

df_movies.head()

"""# Singular Value Decomposition"""

df_ratings=df_ratings[:2000000] #take the first 2 million ratings
#create MxN matrix
df_movie_features = df_ratings.pivot(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0) #NaN values are filled with 0.0

df_movie_features.head()

A = df_movie_features.to_numpy()
user_ratings_mean = np.mean(A, axis = 1)
centered = A - user_ratings_mean.reshape(-1, 1)

"""## **A = USV^T**"""

# 
U, singular_values, V_T = svds(centered, k = 50)

U

V_T

singular_values = np.diag(singular_values)

singular_values

# predict movie ratings
predicted_ratings = np.dot(np.dot(U, singular_values), V_T) + user_ratings_mean.reshape(-1, 1)

preds_df = pd.DataFrame(predicted_ratings, columns = df_movie_features.columns)
preds_df.head()

"""# Recommendations"""

def recommend_movies(preds_df, userId, movies_df, ratings_df, n=5):
    
    # Get Predictions
    user_row = userId - 1 #account for 0 index
    user_predictions = preds_df.iloc[user_row].sort_values(ascending=False) 
    # Merge Movie Titles
    user_data = ratings_df[ratings_df.userId == (userId)]
    watched = (user_data.merge(movies_df, how = 'left', on = 'movieId').
                     sort_values(['rating'], ascending=False))
    # Recommend
    recommendations = (movies_df[~movies_df['movieId'].isin(watched['movieId'])]).merge(pd.DataFrame(user_predictions).reset_index(), how = 'left', on= 'movieId'
                                                                                ).rename(columns = {user_row: 'Predictions'}
                                                                                ).sort_values('Predictions', ascending = False
                                                                                ).iloc[:n, :-1]
                      

    return watched, recommendations

watched, recommendations = recommend_movies(preds_df, 3000, df_movies, df_ratings, 15)

# movies the user has watched an rated
watched.head(10)

# top n recommendations based on user ratings
recommendations

"""# Correlation


"""

#matrix = df_movie_features
df_movies['title'] = df_movies['title'].apply(lambda old_string: old_string.strip())
df = pd.merge(df_movies, df_ratings, on='movieId', how='inner')
df.head()

#create sparse MxN matrix
search_matrix = df.pivot_table(
    index='userId',
    columns='title',
    values='rating'
).fillna(0) #NaN values are filled with 0.0

"""## Check Data"""

#creating average mean rating of movie_title
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
#adding number of ratings on movie
ratings['number_of_ratings'] = df.groupby('title')['rating'].count()
ratings = ratings.reset_index()
ratings.head()

#most rated movie
ratings['title'] = ratings['title'].apply(lambda old_string: old_string.strip())
ratings.sort_values('number_of_ratings', ascending=False).head(10)

"""##Making Recommendation Similar to "Input""""

#Making Recommendation by input
input='Rain Man (1988)'
input_rating = search_matrix[input]

#Finding similar movie
similar_to_input = search_matrix.corrwith(input_rating).sort_values(ascending=False)
similar_to_input.head()

"""##Threshold for min num of ratings"""

#create dataframe
sim_input = pd.DataFrame(similar_to_input, columns=['Correlation'])
sim_input = pd.merge(sim_input, df_movies, on='title', how='left')
sim_input.dropna(inplace=True)
sim_input.head()

#adding in ratings
# sim_input = sim_input.join(ratings['number_of_ratings'])
sim_input = pd.merge(sim_input, ratings, on='title', how='left')
sim_input.head()

sim_input[sim_input['number_of_ratings']>500].sort_values(by='Correlation',ascending=False).head(10)
