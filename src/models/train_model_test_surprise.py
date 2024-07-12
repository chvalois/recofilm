import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
import time

from src.data.db.database_functions import get_engine, sql
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import KFold, GridSearchCV    


def train_model_svd():

    # Start timer
    start_time = time.time()

    # Use SQL Alchemy engine
    engine, inspector = get_engine()

    query = "SELECT * FROM ratings LIMIT 10000;"
    df = pd.read_sql(query, engine)
    df = df.drop(columns = {'created_at'})
    print(df.head())

    query_time = time.time()
    # Calculate elapsed time
    elapsed_time = query_time - start_time
    print("Query took: ", round(elapsed_time, 4), "seconds")

    # A reader is still needed but only the rating_scale param is required.
    reader = Reader(rating_scale=(1, 5))

    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[["user_id", "movie_id", "rating"]], reader)

    param_grid = {"n_epochs": [5, 10], "lr_all": [0.002, 0.005], "reg_all": [0.4, 0.6]}
    gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)

    gs.fit(data)

    # best RMSE score
    print("Best Score RMSE", gs.best_score["rmse"])

    # combination of parameters that gave the best RMSE score
    print("Best Paramters", gs.best_params["rmse"])

    gridsearch_time = time.time()
    # Calculate elapsed time
    elapsed_time = gridsearch_time - query_time
    print("Grid Search took: ", round(elapsed_time, 4), "seconds")

    # We can now use the algorithm that yields the best rmse:
    algo = gs.best_estimator["rmse"]
    algo.fit(data.build_full_trainset())

    train_time = time.time()
    # Calculate elapsed time
    elapsed_time = train_time - gridsearch_time
    print("Training model took: ", round(elapsed_time, 4), "seconds")


    # End timer
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Total time: ", round(elapsed_time, 4), "seconds")

train_model_svd()