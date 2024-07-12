import pandas as pd
import os
import sys
sys.path.append('src')
from joblib import Memory
from sklearn.feature_extraction.text import TfidfVectorizer


cachedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../cache'))
os.makedirs(cachedir, exist_ok=True)
memory = Memory(cachedir, verbose=True)

@memory.cache
def calculer_matrice_tfidf(df='../data/interim/movies_tags.csv'):
    """
    This function takes a DataFrame containing movie tags and returns the corresponding TF-IDF matrix.
    
    Args:
    - df : pandas DataFrame containing movie tags (tags = genres+tags)
    
    Returns:
    - tfidf_matrix : TF-IDF matrix of movie tags
    """

    df.dropna(subset=['all_tags'], inplace=True)
    
    # Initialize the TF-IDF vectorizer
    tfidf = TfidfVectorizer()

    # Calculate the TF-IDF matrix
    matrice_tfidf = tfidf.fit_transform(df['all_tags'])

    return matrice_tfidf

