from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from datetime import datetime
from typing import List, Optional
import asyncio
import os
from pydantic import BaseModel
import sys
from api_directory.generate_token import users, JWTBearer, decode_jwt, sign_jwt
sys.path.append('../src')
from src.models.content_predict import indices
from src.models.collab_predict import collab_reco
from src.models.hybrid_predict import hybride_reco
from src.models.train_model_svd import load_svd_model
from src.models.genres_predict import get_genre_recommendations, all_genres

# Cration of logger object
log_file_path = './api_directory/logs/api_log.log'

log_dir = os.path.dirname(log_file_path)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

app = FastAPI(
    title="Movie Recommandation's API",
    description="API powered by FastAPI",
    version="1.2.0", 
    openapi_tags=[
    {
        'name':'Authentication',
        'description':'functions used to authenticate as a user'
    },
    {
        'name':'Recommandations',
        'description': 'functions returning a recommandation'
    }])

# Add CORS to API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

svd_model = None

def load_svd_model_sync():
    return load_svd_model() 

async def load_models_and_data():
    global svd_model
    svd_model = await asyncio.to_thread(load_svd_model_sync)

# Load model when API start
@app.on_event("startup")
async def startup_event():
    await load_models_and_data()

@app.get("/")
async def read_root():
    with open(log_file_path, 'a') as file:
        file.write("Root endpoint accessed\n")
    model_status = "Model Loaded" if svd_model else "Model Not Loaded"
    return {"model_status": model_status}

# Log Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.utcnow()
    user_id = None

    # Extract user_id from request
    if request.method in ["POST", "PUT"]:
        try:
            request_body = await request.json()
            user_id = request_body.get("user_id")
        except Exception:
            pass
    elif request.method == "GET":
        user_id = request.query_params.get("user_id")

    # Extract user_id from JWT if not found in request
    if not user_id:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                payload = decode_jwt(token)
                user_id = payload.get("user_id")
            except Exception:
                pass

    response = await call_next(request)

    process_time = datetime.utcnow() - start_time

    with open(log_file_path, 'a') as file:
        file.write(f"{start_time.isoformat()} - user_id: {user_id}, "
                   f"path: {request.url.path}, "
                   f"method: {request.method}, "
                   f"status_code: {response.status_code}, "
                   f"process_time: {process_time.total_seconds()}\n")

    return response

# Object containing JWTBearer class from api_directory/generate_token.py
jwt_bearer = JWTBearer(HTTPBearer)

# Definition of BaseModel class
class UserLogin(BaseModel):
    ''' User Id available in dataset '''
    user_id: int

class HybridRecoRequest(BaseModel):
    ''' Movie title available in dataset '''
    titre: str

class GenresInput(BaseModel):
    genre1: str
    genre2: str
    genre3: str
    excluded_genres: Optional[List[str]] = None


@app.post("/login", name='Generate Token', tags=['Authentication'])
async def login(user_data: UserLogin):
    """
    Description:
    This endpoint allows a user to log in by providing login details : user_id. If the details are valid, it returns a JWT token. Otherwise, an error is raised.

    Args:
    - user_data (UserLogin) : user details to log in.

    Returns:
    - str : a JWT token if the login is successful.

    Raises:
    - HTTPException(404, detail="User not found"): If details, user_id is not recognized, an HTTP 404 exception is raised.
    """
    
    # Extract user_id from class UserLogin
    user_id = user_data.user_id
    
    # Looking for user_id in list users defined in api_directory/genrate_token.py
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Create a new token for the user_id with the function sign_jwt from generate_token.py
    token = sign_jwt(user_id)

    return {"access_token": token}

@app.get("/welcome", name='Logged', tags=['Authentication'])
async def welcome(user_id: int = Depends(jwt_bearer)):
    """
    Description:
    This endpoint returns a message "Welcome {user_id}" only if user is authenticated with a JWT token.

    Args:
    - user_id (int, dependency) : the user_id extracted from the payload of the JWT token sent.

    Returns:
    - JSON : returns a JSON with a welcoming message if user is authenticated.

    Raises:
    - HTTPException(403, details = ["Invalid authentication scheme.", "Invalid token or expired token.", "Invalid authorization code."]): If the token is not valid and the user cannot be authenticated.
    """

    return {"message": f"Welcome {user_id}"}

@app.get("/recommendations", name='Collaborative Filtering Recommandations', tags=['Recommandations'])
async def get_recommendations(user_id: int = Depends(jwt_bearer)):
    """
    Description:
    This endpoint retrieves personalized movie recommendations for the authenticated user based on collaborative filtering.

    Args:
    - user_id (int, dependency) : the user_id extracted from the payload of the JWT token sent.

    Returns:
    - JSON: returns a JSON object containing a list of personalized movies recommendations for the user.
    
    Raises:
    - HTTPException(403, details = ["Invalid authentication scheme.", "Invalid token or expired token.", "Invalid authorization code."]): If the token is not valid and the user cannot be authenticated.
    """

    try:
        recommendations = collab_reco(user_id, svd_model)  # Utilise le modèle chargé globalement
        print(recommendations)
        titles = recommendations['title']
        return {"user_id": user_id, "recommendations": titles.tolist()}  # Conversion en dictionnaire pour JSON
    except KeyError as e:
        raise HTTPException(status_code=404, detail="KeyError: Required data not found.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail="ValueError: Invalid input data.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 


@app.post("/hybrid", name='Hybrid Filtering Recommandations', tags=['Recommandations'])
async def hybrid_reco(request: HybridRecoRequest, user_id: int = Depends(jwt_bearer)):
    """
    Description:
    This endpoint retrieves personalized movie recommendations for the authenticated user based on content-based filtering.

    Args:
    - request (ContentRecoRequest, body): The request object containing the movie title and similarity matrix type.
    - user_id (int, dependency): The user_id extracted from the payload of the JWT token sent.

    Returns:
    - JSON: Returns a JSON object containing a list of personalized movie recommendations for the user. The movies returned should be similar to the one sent in the request body.

    Raises:
    - HTTPException(403, details=["Invalid authentication scheme.", "Invalid token or expired token.", "Invalid authorization code."]): If the token is not valid and the user cannot be authenticated.
    """

    # Check if the movie title is recognized
    if request.titre not in indices.index:
        raise HTTPException(status_code=404, detail="Unknown movie title.")
    try:
        recommendations = hybride_reco(user_id, svd_model, request.titre) 
        titles = recommendations['title']
        return {"user_id": user_id, "recommendations": titles.tolist()}  # Conversion en dictionnaire pour JSON
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 


@app.post("/genre_recommendations", tags=['Recommandations'])
async def genre_recommendations(genres: GenresInput, user_id: int = Depends(jwt_bearer)):
    """
    Description:
    This endpoint retrieves personalized movie recommendations for the authenticated user based on specified genres and exclusions. Could be used to prevent the cold start.

    Args:
    - genres (GenresInput, body): The request object containing the three preferred genres and optional excluded genres.
    - user_id (int, dependency): The user_id extracted from the payload of the JWT token sent for authentication.

    Returns:
    - JSON: Returns a JSON object containing the user ID and a list of movie recommendations that match the criteria.

    Raises:
    - HTTPException(404, detail="Genre '{genre}' is not known."): If any of the provided genres or excluded genres are not in the list of known genres.
    - HTTPException(500, detail=str(e)): For any other internal server errors.
    """
    try:
        # Validate genres
        for genre in [genres.genre1, genres.genre2, genres.genre3]:
            if genre.lower() not in all_genres:
                raise HTTPException(status_code=404, detail=f"Genre '{genre}' is not known.")
        
        if genres.excluded_genres:
            for excluded_genre in genres.excluded_genres:
                if excluded_genre.lower() not in all_genres:
                    raise HTTPException(status_code=404, detail=f"Excluded genre '{excluded_genre}' is not known.")

        # Retrieve recommendations
        recommendations = get_genre_recommendations(user_id,genres.genre1,genres.genre2,genres.genre3,genres.excluded_genres)
        titles = recommendations['title']
        return {"user_id": user_id, "recommendations": titles.tolist()}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))