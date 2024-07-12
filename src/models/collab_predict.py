import pandas as pd
import os
import mlflow
import sys
import time
from joblib import Memory

sys.path.append('src')
from src.models.train_model_svd import load_svd_model
from src.data.db.database_functions import get_engine

cachedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../cache'))
os.makedirs(cachedir, exist_ok=True)
memory = Memory(cachedir, verbose=True)

def collab_reco(user_id, svd_model, num_recommendations=10):
    """
    Description:
    This function generates collaborative movie recommendations for a given user using the provided SVD model.

    Args:
    - user_id (str): The ID of the user for whom recommendations are generated.
    - svd_model (Surprise SVD model): The trained SVD model used for generating recommendations.
    - num_recommendations (int): The number of recommendations to generate. Defaults to 10.
    - start_index (int): The starting index for recommendations, useful for printing the next movies. Defaults to 0.

    Returns:
    - DataFrame: A DataFrame containing the recommended movie titles and their estimated ratings.
    """

    # Check if temporary directory exist otherwise it's created
    temp_dir = "src/models/temp/"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # MLflow experiment
    experiment_name = "SVD_Collab_Pred"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="collab_reco"):

        # Use SQL Alchemy engine
        engine, inspector = get_engine()

        # Generate a list of movies unrated by the user
        query = f"""WITH movies_rated AS (SELECT DISTINCT(movie_id) FROM ratings WHERE user_id={user_id} ORDER BY movie_id)
        SELECT movie_id                                                                                    
        FROM movies                                                                                        
        WHERE movie_id NOT IN (SELECT movie_id FROM movies_rated);"""

        unrated_movies = pd.read_sql(query, engine)
        unrated_movies_list = unrated_movies['movie_id'].to_list()

        avg_rating = 3.525529
        anti_testset = [(user_id, movie_id, avg_rating) for movie_id in unrated_movies_list]
        
        # Generate predictions using the SVD model
        predictions_svd = svd_model.test(anti_testset)
        predictions_svd = pd.DataFrame(predictions_svd)
              
        # Get the titles of the movies
        query = "SELECT * FROM movies;"
        df_movies = pd.read_sql(query, engine)
        movieId_title_map = df_movies.set_index('movie_id')['title'].to_dict()
        predictions_svd['title'] = predictions_svd['iid'].map(movieId_title_map)

        # Rename and reorder the columns for clarity
        predictions_svd = predictions_svd.rename(columns={'uid': 'userId', 'est': 'note'})
        predictions_svd = predictions_svd[['userId', 'title', 'note']]

        # Sort the predictions by rating in descending order
        predictions_svd.sort_values('note', ascending=False, inplace=True)

         # Filter movies with predicted rating >= 4 to form the pool
        pool = predictions_svd[predictions_svd['note'] >= 4]

        # If the pool has fewer than the desired number of recommendations, adjust accordingly
        num_recommendations = min(num_recommendations, len(pool))

        # Select a random sample of the pool
        recommendations = pool.sample(num_recommendations)

        # Sort the predictions by rating in descending order
        recommendations.sort_values('note', ascending=False, inplace=True)

        # Enregistrer le fichier CSV localement
        collab_pred_path = os.path.join(temp_dir, "collab_pred.csv")
        recommendations.to_csv(collab_pred_path, index=False)

        # Log predictions
        mlflow.log_param("user_id", user_id)
        mlflow.log_param("num_recommendations", num_recommendations)
        mlflow.log_artifact(collab_pred_path)

    return recommendations

if __name__ == "__main__":

    # Start timer
    start_time = time.time()

    svd_model = load_svd_model()
    print("model loaded")
    loading_model_time = time.time()
    elapsed_time = loading_model_time - start_time
    print("Loading model took: ", round(elapsed_time, 4), "seconds")

    # test of generate_new_recommendtaions
    user_id = 1000
    recommendations = collab_reco(user_id, svd_model)
    print(recommendations)