from py2neo import Graph, Node, Relationship, authenticate
import pandas as pd
import numpy as np
import math
import credentials

encoded = ""
authenticate("localhost:7474", credentials.user, credentials.password)
graph = Graph('http://localhost:7474/db/data/')
query = ("MATCH (m1:Movie) WITH count(m1) as countm RETURN countm")

query_no_users = ( 'MATCH (u1:`User`)'
        'RETURN u1')

query_no_movies = ('MATCH (m1:`Movie`)'
        'WITH count(m1) as count_movies '
        'RETURN count_movies')

query_no_ratings_by_user = ('MATCH (u1:`User`)-[:`RATED`]->(m1:`Movie`) '
                            'WITH u1,count(m1) AS number_rated_movies  '
                            'RETURN sum(number_rated_movies) as total_ratings')

query_ratings = ('MATCH (u:`User`)-[r:`RATED`]->(m:`Movie`)'
                 'RETURN u.id as user_id, m.id as movie_id, '
                 'm.title as movie_title, r.rating as rating LIMIT 2000')

ratings_df = pd.DataFrame()

user_movie_ratings = graph.cypher.execute(query_ratings)
user_id = list()
movie_id = list()
movie_name = list()
ratings = list()

for record in user_movie_ratings:
    user_id.append(record[0])
    movie_id.append(record[1])
    movie_name.append(record[2])
    ratings.append(record[3])

ratings_df['user_id'] = user_id
ratings_df['movie_id'] = movie_id
ratings_df['movie_name'] = movie_name
ratings_df['ratings'] = pd.Series(ratings)

ratings_df['ratings'] = pd.to_numeric(ratings_df['ratings'])

# print(ratings_df)


class MF():

    # Initializing the user-movie rating matrix, no. of latent features, alpha and beta.
    def __init__(self, R, K, alpha, beta, iterations):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    # Initializing user-feature and movie-feature matrix
    def train(self):
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initializing the bias terms
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # List of training samples
        self.samples = [
        (i, j, self.R[i, j])
        for i in range(self.num_users)
        for j in range(self.num_items)
        if self.R[i, j] > 0
        ]

        # Stochastic gradient descent for given number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 20 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process

    # Computing total mean squared error
    def mse(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    # Ratings for user i and moive j
    def get_rating(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    # Full user-movie rating matrix
    def full_matrix(self):
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)


def perdict_rating(ratings_df, user_id, movie_id):
    ratings_df_full_pivot = ratings_df.pivot(index = 'user_id', columns ='movie_id', values = 'ratings').fillna(0)

    # We choose the user with the id 2 giving a rating of 3 to the movie with the id 356 - Forresr Gump
    user_of_interest_index =\
        ratings_df[(ratings_df['movie_id']==movie_id) & (ratings_df['user_id']==user_id)].index
    # We eliminate that particular row and try to predict the rating of user with id 2 for the same movie
    ratings_df_dr = ratings_df.drop(user_of_interest_index)

    # create pivot table for the algorithm after aliminating the rating of interest
    ratings_df_pivot = ratings_df_dr.pivot(index = 'user_id', columns ='movie_id', values = 'ratings').fillna(0)
    R= np.array(ratings_df_pivot)
    mf = MF(R, K=20, alpha=0.001, beta=0.01, iterations=100)

    #print(ratings_df_pivot)
    print("---------------------")
    print("Rating given by user with id" , user_id, "to the movie with the id " ,movie_id, ": ")
    print(ratings_df_full_pivot.ix[user_id, movie_id])

    print("---------------------")
    # train the model
    training_process = mf.train()
    print("---------------------")
    print("PREDICTED rating given by user with id" , user_id, "to the movie with the id " ,movie_id, ": ")
    print(mf.get_rating(user_id, movie_id))

    print("---------------------")
    print("MSE calculated: ")
    print(mf.mse())

    print("---------------------")
    print("RMSE calculated: ")
    print(math.sqrt(mf.mse()))

perdict_rating(ratings_df = ratings_df,
               user_id = 2,
               movie_id = 356)