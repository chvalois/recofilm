from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import pandas as pd
import sys
sys.path.append('src')
import os
from src.features.build_tfidf_matrix import calculer_matrice_tfidf


def is_running_under_pytest():
    return 'pytest' in sys.modules

if is_running_under_pytest():
    df_content_tags = pd.read_csv("./tests/fixtures/movies_tags_test.csv")   # pour les tests
else:
    df_content_tags = pd.read_csv("./src/data/interim/movies_tags.csv")

# Create an index series
indices = pd.Series(range(0, len(df_content_tags)), index=df_content_tags.title)

def content_based_reco(titre, num_recommendations=10):
    """
    Description:
    This function generates content-based movie recommendations based on the provided movie title and similarity matrix.

    Args:
    - titre (str): The title of the movie for which recommendations are generated.
    - num_recommendations (int): The number of recommendations to generate. Defaults to 10.

    Returns:
    - dict: A dictionary containing movie titles as keys and their similarity scores as values.
    """
    # Check if temporary directory exist otherwise it's created
    temp_dir = "src/models/temp/"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # MLflow experiment
    experiment_name = "Content_Pred"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="content_reco"):
        # Calculate TF-IDF matrix calculate_matrix_tfidf is in src/features/build_tfidf_matrix.py
        matrice_tfidf = calculer_matrice_tfidf(df_content_tags)

        # Get the index of the provided movie title
        idx = indices[titre]

        # Calculate similarity matrices
        sim_cos = cosine_similarity(matrice_tfidf, matrice_tfidf)   

        # Calculate similarity scores between the provided movie and all other movies and sort in descsending order
        scores_similarite = list(enumerate(sim_cos[idx]))
        scores_similarite = sorted(scores_similarite, key=lambda x: x[1], reverse=True)

        # Select the top similar movies excluding the provided movie itself
        top_similar = scores_similarite[1:num_recommendations+1]

        # Create a dictionary containing recommended movie titles and their similarity scores
        recommendations = [(indices.index[idx], score) for idx, score in top_similar]
        recommendations = pd.DataFrame(recommendations)
        recommendations = recommendations.rename(columns={0: 'title', 1: 'score'})

        # Save into a temporary csv file
        content_pred_path = os.path.join(temp_dir, "content_pred.csv")
        recommendations.head(num_recommendations).to_csv(content_pred_path, index=False)
       
        # Log predictions in MLflow
        mlflow.log_param("film", titre)
        mlflow.log_param("num_recommendations", num_recommendations)
        mlflow.log_artifact(content_pred_path)

    return recommendations

if __name__ == "__main__":
    recommendations = content_based_reco('Toy Story (1995)')
    print(recommendations)

