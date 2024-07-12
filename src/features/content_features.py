import os
import pandas as pd
import re
from src.data.db.database_functions import get_engine, sql

def load_data():
    """
    Charge les données à partir des fichiers CSV.

    Returns
    -------
    tuple
        Un tuple contenant les DataFrames des tags, des films, des tags de genome et des scores de genome.
    """
    df_tags = pd.read_csv('../data/raw/tags.csv')
    df_movies = pd.read_csv('../data/raw/movies.csv')
    df_genome_tags = pd.read_csv('../data/raw/genome-tags.csv')
    df_genome_scores = pd.read_csv('../data/raw/genome-scores.csv')
    return df_tags, df_movies, df_genome_tags, df_genome_scores

def load_data_from_db():
    """
    Charge les données à partir de la base postgres.

    Returns
    -------
    tuple
        Un tuple contenant les DataFrames des tags, des films, des tags de genome et des scores de genome.
    """
     # Use SQL Alchemy engine
    engine, inspector = get_engine()

    query = "SELECT * FROM tags;"
    df_tags = pd.read_sql(query, engine)

    query = "SELECT * FROM movies;"
    df_movies = pd.read_sql(query, engine)

    query = "SELECT * FROM genome_tags;"
    df_genome_tags = pd.read_sql(query, engine)    
    
    query = "SELECT * FROM genome_scores;"
    df_genome_scores = pd.read_sql(query, engine)

    return df_tags, df_movies, df_genome_tags, df_genome_scores


def select_top_relevant_tags(df_genome_tags, df_genome_scores):
    """
    Sélectionne les tags les plus pertinents pour chaque film.

    Parameters
    ----------
    df_genome_tags : pd.DataFrame
        DataFrame contenant les tags de genome.
    df_genome_scores : pd.DataFrame
        DataFrame contenant les scores de genome.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les tags les plus pertinents pour chaque film.
    """
    # df_tags_relevance = pd.merge(df_genome_tags, df_genome_scores, on='tagId', how='outer')
    # df_top_relevance = df_tags_relevance.groupby('movieId').apply(lambda x: x.nlargest(3, 'relevance')).reset_index(drop=True)
    # df_top_relevance_grouped = df_top_relevance.groupby('movieId')['tag'].apply(lambda x: ', '.join(x)).reset_index()
    df_tags_relevance = pd.merge(df_genome_tags, df_genome_scores, on='gtag_id', how='outer')
    df_top_relevance = df_tags_relevance.groupby('movie_id').apply(lambda x: x.nlargest(3, 'relevance')).reset_index(drop=True)
    df_top_relevance_grouped = df_top_relevance.groupby('movie_id')['tag'].apply(lambda x: ', '.join(x)).reset_index()
    df_top_relevance_grouped.rename(columns={'tag': 'tags'}, inplace=True)
    return df_top_relevance_grouped

def merge_tags(df_tags):
    """
    Regroupe les tags par movieId.

    Parameters
    ----------
    df_tags : pd.DataFrame
        DataFrame contenant les tags.

    Returns
    -------
    pd.DataFrame
        DataFrame regroupant les tags par movieId.
    """
    df_tags['tag'] = df_tags['tag'].astype(str)
    # df_tags_grouped = df_tags.groupby('movieId')['tag'].apply(lambda x: ', '.join(x)).reset_index()
    df_tags_grouped = df_tags.groupby('movie_id')['tag'].apply(lambda x: ', '.join(x)).reset_index()
    df_tags_grouped.rename(columns={'tag': 'tags'}, inplace=True)
    return df_tags_grouped

def merge_data(df_top_relevance_grouped, df_tags_grouped, df_movies):
    """
    Fusionne les données des tags pertinents et des tags groupés avec les données des films.

    Parameters
    ----------
    df_top_relevance_grouped : pd.DataFrame
        DataFrame contenant les tags les plus pertinents pour chaque film.
    df_tags_grouped : pd.DataFrame
        DataFrame regroupant les tags par movieId.
    df_movies : pd.DataFrame
        DataFrame contenant les données des films.

    Returns
    -------
    pd.DataFrame
        DataFrame fusionné.
    """
    # df_total_tags = pd.merge(df_top_relevance_grouped, df_tags_grouped, on='movieId', how='outer')
    df_total_tags = pd.merge(df_top_relevance_grouped, df_tags_grouped, on='movie_id', how='outer')

    df_total_tags['tags_x'] = df_total_tags['tags_x'].astype(str)
    df_total_tags['tags_y'] = df_total_tags['tags_y'].astype(str)
    df_total_tags['tags'] = df_total_tags['tags_x'] + ', ' + df_total_tags['tags_y']
    df_total_tags.drop(['tags_x', 'tags_y'], axis=1, inplace=True)
    # df_total_tags = pd.merge(df_movies, df_total_tags, on='movieId', how='outer')
    df_total_tags = pd.merge(df_movies, df_total_tags, on='movie_id', how='outer')

    df_total_tags['genres'] = df_total_tags['genres'].str.replace('|', ', ')
    df_total_tags['genres'] = df_total_tags['genres'].astype(str)
    df_total_tags['tags'] = df_total_tags['tags'].astype(str)
    df_total_tags['all_tags'] = df_total_tags['genres'] + ', ' + df_total_tags['tags']
    df_total_tags.drop(['genres', 'tags'], axis=1, inplace=True)
    return df_total_tags

def clean_tags(tags_list):
    """
    Nettoie les tags en ne conservant que ceux composés uniquement de lettres et de tirets.

    Parameters
    ----------
    tags_list : list
        Liste de tags.

    Returns
    -------
    list
        Liste nettoyée de tags.
    """
    clean_tags_list = []
    for tag in tags_list:
        if not isinstance(tag, str):
            continue
        cleaned_tag = tag.strip().lower()
        if cleaned_tag == 'nan':
            continue
        if re.match(r'^[a-zA-Z\-]+$', cleaned_tag):
            clean_tags_list.append(cleaned_tag)
    clean_tags_list = list(set(clean_tags_list))
    return clean_tags_list


def process_df(df_total_tags):
    """
    Traite les tags et sauvegarde les données nettoyées.

    Parameters
    ----------
    df_total_tags : pd.DataFrame
        DataFrame avec les genres et les tags traités.
    """
    df_total_tags['all_tags'] = df_total_tags['all_tags'].apply(lambda x: x.split(','))
    df_total_tags['all_tags'] = df_total_tags['all_tags'].apply(clean_tags)
    df_total_tags['all_tags'] = df_total_tags['all_tags'].apply(lambda x: ', '.join(x))
    return df_total_tags

if __name__ == "__main__":
    """
    Fonction principale pour exécuter toutes les étapes du traitement des données.
    """
#    df_tags, df_movies, df_genome_tags, df_genome_scores = load_data()
    df_tags, df_movies, df_genome_tags, df_genome_scores = load_data_from_db()

    df_top_relevance_grouped = select_top_relevant_tags(df_genome_tags, df_genome_scores)
    df_tags_grouped = merge_tags(df_tags)
    df_total_tags = merge_data(df_top_relevance_grouped, df_tags_grouped, df_movies)
    df_total_tags = process_df(df_total_tags)

    # Check if directory exist otherwise it's created
    interim_dir = "src/data/interim/"
    if not os.path.exists(interim_dir):
        os.makedirs(interim_dir)

    # Vérification de l'existence du fichier movies_tags.csv
    file_path = os.path.join(interim_dir, 'movies_tags.csv')
    if os.path.exists(file_path):
        os.remove(file_path)
    
    df_total_tags.to_csv(file_path, index=False)
