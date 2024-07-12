import pandas as pd
import mlflow
import mlflow.sklearn
from surprise import SVD
from surprise.model_selection import cross_validate
import logging
import pickle
import numpy as np
from joblib import Memory
import os
import time
import sys

sys.path.append('src')
from src.models.load_svd_data import load_and_prepare_data_from_db

cachedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../cache'))
os.makedirs(cachedir, exist_ok=True)
memory = Memory(cachedir, verbose=True)

logger = logging.getLogger(__name__)

# Define experiment name
experiment_name = "SVD_Movie_Reco"
mlflow.set_experiment(experiment_name)


def evaluate_svd_model(measures=['rmse', 'mae'], cv=5):
    """
    Description:
    This function evaluates an SVD model using cross-validation on the provided dataset. 
    It calculates specified performance measures and returns the trained SVD model along with the cross-validation results.

    Args:
    - measures (list): A list of performance measures to evaluate. Defaults to ['RMSE', 'MAE'].
    - cv (int): The number of cross-validation folds. Defaults to 5.

    Returns:
    - SVD: The trained SVD model.
    - dict: A dictionary containing cross-validation results.
    """
    df_surprise,_ = load_and_prepare_data_from_db()
    svd = SVD(n_factors=100, n_epochs=30, lr_all=0.01, reg_all=0.05)
    
    with mlflow.start_run(run_name="evaluation"):
        cv_results = cross_validate(svd, df_surprise, measures=measures, cv=cv, verbose=True)
        # Log metrics for each measure
        for metric in measures:
            mean_metric = cv_results[f'test_{metric}'].mean()
            std_metric = cv_results[f'test_{metric}'].std()
            mlflow.log_metric(f'{metric}_mean', mean_metric)
            mlflow.log_metric(f'{metric}_std', std_metric)
        
        # Log the training and testing times
        mlflow.log_metric('fit_time_mean', np.mean(cv_results['fit_time']))
        mlflow.log_metric('fit_time_std', np.std(cv_results['fit_time']))
        mlflow.log_metric('test_time_mean', np.mean(cv_results['test_time']))
        mlflow.log_metric('test_time_std', np.std(cv_results['test_time']))

        mlflow.log_params({"n_factors": 100, "n_epochs": 30, "lr_all": 0.01, "reg_all": 0.05})
        mlflow.log_params({"measures": measures, "cv": cv})
    #return svd, cv_results


def train_svd_model():
    """
    Description:
    This function trains an SVD model on the provided dataset and saves the trained model to a pickle file. 
    It returns the trained SVD model.

    Args:
    None

    Returns:
    - SVD: The trained SVD model.
    """

        # Start timer
    start_time = time.time()

    # Load and Prepare Data
    _, train_set = load_and_prepare_data_from_db()

    # Get the global mean rating
    moyenne = train_set.global_mean
    logger.info(moyenne)

    # Create the mapping from raw user IDs to inner user IDs
    raw_to_inner_uid_mapping = {train_set.to_raw_uid(inner_uid): inner_uid for inner_uid in train_set.all_users()}

    # Print the mappings
    # print("\nRaw to Inner User ID Mapping:")
    # for raw_uid, inner_uid in raw_to_inner_uid_mapping.items():
    #     print(f"Raw User ID: {raw_uid}, Inner User ID: {inner_uid}")

    # Create the mapping from raw movie IDs to inner movie IDs
    raw_to_inner_iid_mapping = {train_set.to_raw_iid(inner_iid): inner_iid for inner_iid in train_set.all_items()}

    # Print the mappings
    #print("\nRaw to Inner Item ID Mapping:")
    #for raw_iid, inner_iid in raw_to_inner_iid_mapping.items():
    #    print(f"Raw Item ID: {raw_iid}, Inner Item ID: {inner_iid}")

    user_id_mapping = pd.DataFrame(list(raw_to_inner_uid_mapping.items()), columns=['Raw User ID', 'Inner User ID'])
    user_id_mapping.to_csv('src/models/user_id_mapping.csv', index = None)

    item_id_mapping = pd.DataFrame(list(raw_to_inner_iid_mapping.items()), columns=['Raw Item ID', 'Inner Item ID'])
    item_id_mapping.to_csv('src/models/item_id_mapping.csv', index = None)

    load_data_time = time.time()
    elapsed_time = load_data_time - start_time
    logger.info(f"Loading data took: {round(elapsed_time, 4)} seconds")

    # Train SVD Model
    svd_model = SVD(n_factors=100, n_epochs=30, lr_all=0.01, reg_all=0.05).fit(train_set)

    training_svd_time = time.time()
    elapsed_time = training_svd_time - load_data_time
    logger.info(f"Training data took: {round(elapsed_time, 4)} seconds")

    # Saving Model as pkl file
    model_path = "src/models/svd_model.pkl"
    with open(model_path, "wb") as filehandler:
        pickle.dump(svd_model, filehandler)

    saving_model_time = time.time()
    elapsed_time = saving_model_time - training_svd_time
    logger.info(f"Saving model took: {round(elapsed_time, 4)} seconds")

    # Saving Model in MLFlow
    try:
        with mlflow.start_run(run_name="training"):
            mlflow.log_params({"n_factors": 100, "n_epochs": 30, "lr_all": 0.01, "reg_all": 0.05})
            mlflow.sklearn.log_model(svd_model, "svd_model")
            mlflow.log_artifact(model_path)
    except Exception as e:
        # Handle the exception
        logger.info(f"An error occurred with MLFlow model saving: {e}")
        raise  # Re-raise the exception to mark the task as failed

    saving_mlflow_model_time = time.time()
    elapsed_time = saving_mlflow_model_time - saving_model_time
    logger.info(f"Saving model in ML Flow took: {round(elapsed_time, 4)} seconds")

    # return svd_model

@memory.cache
def load_svd_model(filepath="./src/models/svd_model.pkl"):
    """
    Description:
    This function loads a previously trained SVD model from a file and returns it.

    Args:
    None

    Returns:
    - SVD: The loaded SVD model.
    """
    
    with open(filepath, "rb") as filehandler:
        return pickle.load(filehandler)

if __name__ == "__main__":

    try:
        train_svd_model()
        logger.info("Le modèle SVD a été entraîné et sauvegardé avec succès.")
        load_svd_model()
    except Exception as e:
        # Handle the exception
        logger.info(f"An error occurred: {e}")
        raise  # Re-raise the exception to mark the task as failed
