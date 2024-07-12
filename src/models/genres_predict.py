import pandas as pd
import os
import mlflow
import sys


sys.path.append('src')
from src.data.db.database_functions import get_engine

# Use SQL Alchemy engine
engine, inspector = get_engine()

all_genres = ["Action","Adventure","Animation","Children","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","IMAX","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]
all_genres = [genre.lower() for genre in all_genres]

def get_genre_recommendations(user_id, genre1, genre2, genre3, excluded_genres=None, num_recommendations=10):
    # Check if temporary directory exist otherwise it's created
    temp_dir = "src/models/temp/"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # MLflow experiment
    experiment_name = "Genres_Pred"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="genres_reco"):

        query = "SELECT * FROM movies;"
        df_movies = pd.read_sql(query, engine)

        query = "SELECT * FROM ratings;"
        df_ratings = pd.read_sql(query, engine)

        avg_ratings = df_ratings.groupby('movie_id')['rating'].agg(['mean', 'count']).reset_index()
        avg_ratings.columns = ['movie_id', 'avg_rating', 'num_ratings']

        global_mean = avg_ratings['avg_rating'].mean()
        C = global_mean / 2
        avg_ratings['bayesian_avg_rating'] = (C * global_mean + avg_ratings['avg_rating'] * avg_ratings['num_ratings']) / (C + avg_ratings['num_ratings'])

        movies_with_avg_rating = df_movies.merge(avg_ratings[['movie_id', 'bayesian_avg_rating']], on='movie_id', how='left')

        movies_with_avg_rating['genres_list'] = movies_with_avg_rating['genres'].apply(lambda x: [genre.lower() for genre in x.split('|')])
        movies_with_avg_rating = movies_with_avg_rating.drop("genres", axis=1)

        genre1 = genre1.lower()
        genre2 = genre2.lower()
        genre3 = genre3.lower()
        excluded_genres = [genre.lower() for genre in excluded_genres] if excluded_genres else []

        condition_all_genres = movies_with_avg_rating['genres_list'].apply(lambda x: all(genre in x for genre in [genre1, genre2, genre3]))
        if excluded_genres:
                condition_excluded_genres = movies_with_avg_rating['genres_list'].apply(lambda x: not any(excluded_genre in x for excluded_genre in excluded_genres))
        else:
                condition_excluded_genres = True  
        movies_all_genres = movies_with_avg_rating.loc[condition_all_genres & condition_excluded_genres].sort_values("bayesian_avg_rating", ascending=False)

        condition_any_genre = movies_with_avg_rating['genres_list'].apply(lambda x: any(genre in x for genre in [genre1, genre2, genre3]))

        movies_any_genre = movies_with_avg_rating.loc[condition_any_genre & ~condition_all_genres & condition_excluded_genres].sort_values("bayesian_avg_rating", ascending=False)

        num_all_genres = int(num_recommendations * 0.7) 
        num_any_genres = num_recommendations - num_all_genres  

        if len(movies_all_genres) < num_all_genres:
                all_genres_recommendations = movies_all_genres
                num_any_genres = num_recommendations - len(all_genres_recommendations)
        else:
                all_genres_recommendations = movies_all_genres.head(num_all_genres)

        any_genres_recommendations = movies_any_genre.head(num_any_genres)

        # Combine and deduplicate recommendations
        recommendations = pd.concat([all_genres_recommendations, any_genres_recommendations])
        recommendations = recommendations.head(num_recommendations)
        recommendations = recommendations.drop_duplicates(subset=['movie_id'])

        genres_reco_path = os.path.join(temp_dir, "genres_reco.csv")
        recommendations.to_csv(genres_reco_path, index=False)

        # Log predictions
        mlflow.log_param("user_id", user_id)
        mlflow.log_param("num_recommendations", num_recommendations)
        mlflow.log_artifact(genres_reco_path)

        return recommendations


