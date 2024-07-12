Movie's Recommendations 
==============================

This project is a starting Pack for MLOps projects based on the subject "movie_recommandation". 

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── api_directory      <- Scripts to execute the API
    │   ├── logs
    │   │   └── api_log.log     <- Logs from api calls
    │   ├── api.py              <- Main api file
    │   ├── generate_token.py   <- Generate and decode token for authentication
    │   ├── preferences.py      <- Generate top 3 genres for a user
    │   └── requests.txt        <- Few curl requests for api's routes
    │
    ├── cache             
    │
    ├── dockerfiles             <- Dockerfiles used in docker-compose
    │   ├── Dockerfile          <- Dockerfile used to create project image used for Airflow services
    │   ├── Dockerfile_api
    │   └── Dockerfile_mlflow     
    │
    ├── grafana                 
    │   ├── dashboards          <- Contains custom dashboard to monitor Postgres database
    │   └── datasources         <- Setup Prometheus datasource
    │
    ├── MLflow                  <- All mlflow runs and artifacts
    │
    ├── notebooks               <- Jupyter notebooks
    │
    ├── references              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures             <- Generated graphics and figures to be used in reporting
    │
    ├── src                     <- Source code for use in this project
    │   ├── config              <- Rclone configuration file
    │   ├── dags                <- Contains Airflow DAGs
    │   ├── data                <- Scripts to download or generate data
    │   │   ├── db              <- Scripts to generate postgres database
    │   │   │   ├── initialize_database.py   
    │   │   │   ├── drop_database.pgsql                     
    │   │   │   ├── initialize_database.pgsql 
    │   │   │   └── databas_functions.py   
    │   │   ├── interim                     <- Intermediate data that has been transformed
    │   │   ├── processed                   <- The final, canonical data sets for modeling.
    │   │   ├── raw                         <- The original, immutable data dump
    │   │   ├── check_structure.py    
    │   │   ├── import_raw_data.py 
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   ├── build_features.py
    │   │   ├── content_features.py
    │   │   └── build_tfidf_matrix.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions
    │   │   │                 
    │   │   ├── temp       <- Temporary csv files for predictions
    │   │   ├── collab_predict.py                   <- Predictions with collaborative filtering model (SVD)
    │   │   ├── content_predict.py                  <- Predictions with content based model
    │   │   ├── genres_predict.py                   <- Predictions based on genres selection
    │   │   ├── grid_seacrh_svd.py                  <- Grid-Search on SVD model
    │   │   ├── hybrid_predict.py                   <- Predictions with hybrid model
    │   │   ├── item_id_mapping.csv
    │   │   ├── load_svd_data.py                    
    │   │   ├── train_model_svd.py                  <- Train and evaluate SVD model
    │   │   ├── train_model_test_surprise.py
    │   │   └── user_id_mapping.csv
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │         └── visualize.py
    │
    ├── tests               <- Unit tests
    │   ├── fixtures              
    │   ├── test_api        <- Unitest for api
    │   ├── test_data       <- Unitest for database
    │   ├── test_features   <- Unitest for features
    │   └── test_models     <- Unitest for models
    │
    ├── .dockerignore 
    ├── .gitignore 
    ├── .env 
    ├── alertmanager.yml            <- Setup of Prometheus Alerts Management
    ├── docker-compose.yml          <- docker-compose configuration file that contains all dockerized services 
    ├── prometheus_rules.yml        <- Setup of Prometheus Rules to generate Alerts
    ├── prometheus.yml              <- Setup of Prometheus Exporter Jobs
    ├── pytest.ini
    ├── requirements.txt                <- The requirements file for reproducing the analysis environment
    ├── requirements_wo_surprise.txt    <- This requirements file is only used for Github Actions because of 
    │                                   the Scikit Surprise package that does not install properly via pip
    ├── setup.py.txt
    └── setup.sh                    <- The file used to launch the application
------------

## Steps to follow 

Convention : All python scripts must be run from the root specifying the relative file path.

### 1- Create a virtual environment using Virtualenv.

    `python -m venv my_env`

###   Activate it 

    `./my_env/Scripts/activate`

### 2- Ensure you are at the root of the project and run setup.sh to install and start the application

    For MAC users :
    `chmod +x setup_mac.sh`
    `./setup_mac.sh`

    For Windows users using a WSL Linux distribution
    `chmod +x setup_linux_wsl.sh`
    `bash setup_linux_wsl.sh`

    If ever the sh script fails because of "\r" errors, please switch the .sh file to LF instead of CRLF
    in "Select End of Line Sequence" option on the bottom right of VS Code and save it.
    
It may take a few minutes

### 3- Access the different UI on a navigator
    - mlflow on http://localhost:5001
    - api on http://localhost:8000  
    - airflow webserver on http://localhost:8080
        credentials = admin/admin
    - prometheus on http://localhost:9090
    - grafana on http://localhost:3000
        credentials = admin/admin


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


Example of possible requests: 
# /login
`curl -X POST "http://localhost:8000/login" -H "Content-Type: application/json" -d '{"user_id": 123}'`

# /welcome
`curl -X GET "http://localhost:8000/welcome" -H "Authorization: Bearer <token>"`

# /recommendations
`curl -X GET "http://localhost:8000/recommendations" -H "Authorization: Bearer <token>"`

# /hybrid
`curl -X POST "http://localhost:8000/hybrid" -H "Authorization: Bearer <token>" -H "Content-Type: application/json" -d '{"titre": "Toy Story (1995)"}`

# /genres_recommendations
`curl -X POST "http://localhost:8000/genres_recommendations" -H "Authorization: Bearer <token>" -H "Content-Type: application/json" -d '{"genre1": "Animation", "genre2": "Fantasy", "genre3": "Drama", "excluded_genres": ["Horror"]}`
