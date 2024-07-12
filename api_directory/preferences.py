import pandas as pd

def get_user_preferences(user_id, user_matrix_filename):
    """
    Description:
    Get the top 3 movie genres based on user preferences.

    Args:
    - user_id (int): The ID of the user for whom preferences are to be retrieved.
    - user_matrix_filename (str): The filename of the CSV file containing user preferences data.

    Returns:
    - list[str]: A list containing the top 3 movie genres based on the user's preferences.
    """
    # Load user matrix
    user_matrix = pd.read_csv(user_matrix_filename)

    # Filter user preferences by user_id, then select all columns (genres) except the column user_id (column 0)
    user_preferences = user_matrix[user_matrix["userId"] == user_id].iloc[:, 1:]

    # Get top 3 genres based on user preferences
    # Convert the DataFrame to a numpy array, sort the preference values in ascending order then select the last 3 indices
    top_3_genres = user_preferences.columns[user_preferences.values.squeeze().argsort()[-3:]].tolist()

    return top_3_genres
