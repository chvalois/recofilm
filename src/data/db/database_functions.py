from sqlalchemy import create_engine, text, inspect
import pandas as pd
import os
from datetime import datetime

username = os.getenv('POSTGRES_USER')
password = os.getenv('POSTGRES_PASSWORD')
database = os.getenv('POSTGRES_DB')

#host = '127.0.0.1'
#host = 'localhost'
host = 'postgres-db' 
port = '5432'  # default PostgreSQL port is 5432

import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Create the database engine
try:
    engine = create_engine(f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}', echo=True)
    inspector = inspect(engine)
    print('Connection to database succeeded')
except Exception as e:
    print(e)
    print(username, password, database)
    print('Connection to database failed')


def get_engine():
    return engine, inspector


def sql(query):
    with engine.connect() as conn:
        conn.execute(text(query))


def execute_pgsql_query(filename):
    """ Executes a SQL Query in the SQL Alchemy Engine """
    with open(filename, 'r') as file:
        query = file.read()

    # Connect to the database and execute the SQL commands
    with engine.connect() as connection:
        try:
            # Execute each command separately
            for command in query.split(';'):
                if command.strip():  # Skip empty commands
                    connection.execute(text(command))
            print("Tables created successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")



def init_csv_to_sql(table_name, csv_path, mapping):
    """ Feed a table in the postgresDB from a CSV file"""

    print(f'--- TABLE {table_name} ---')

    # Read CSV
    print(f'Reading CSV file for table {table_name}')
    df = pd.read_csv(csv_path)
    
    # Edit and save processed CSV
    print(f'Editing and saving processed CSV file for table {table_name}')
    df = df.rename(columns = mapping)

    if 'created_at' in list(df.columns):
        df['created_at'] = pd.to_datetime(df['created_at'], unit='s')

    if 'tmdb_id' in list(df.columns):
        df['tmdb_id'] = df['tmdb_id'].fillna(-1).astype('int')

    filename = csv_path.split('/')[-1]
    processed_path = f'../processed/{filename}'
    df.to_csv(processed_path, index=False)    

    # Inject CSV data into database
    print(f'Injecting initial data into table {table_name}')
    conn = engine.raw_connection()
    cursor = conn.cursor()

    try:
        with open(processed_path, 'r') as f:
            cursor.copy_expert(f"COPY {table_name} FROM STDIN WITH CSV HEADER", f)
        conn.commit()
        print(f'Injection of table {table_name} succeeded')
    except Exception as e:
        print(e)
        print(f'Injection of table {table_name} failed')
    finally:
        cursor.close()
        conn.close()

def create_initial_users():
    """ Feed the table 'users' in the postgresDB """
    users = pd.read_csv('../raw/ratings.csv')
    users = users.drop(columns = {'movieId', 'rating', 'timestamp'}).rename(columns = {'userId': 'user_id'})
    users = users.drop_duplicates()
    users['user_key'] = users['user_id']        # To be encoded later ?
    #now = datetime.now()
    #users['updated_at'] = now

    users.to_csv('../processed/users.csv', index=False)

    try:
        print(f'Injecting initial data into users')
        users.to_sql("users", engine, if_exists='append', index=False)
    except Exception as e:
        print(e)
        print(f'Injection of table users failed')