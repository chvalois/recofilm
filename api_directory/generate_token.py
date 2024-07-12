import time
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Request, HTTPException
import jwt
import pandas as pd
import secrets
import sys

sys.path.append('../src')
from src.data.db.database_functions import get_engine


# Generate a JWT secret key and define the encryption algorithm
JWT_SECRET = secrets.token_hex(32)
JWT_ALGORITHM = "HS256"


def load_user_ids(dataset):
    """
    Description:
    Load user IDs from a CSV dataset.

    Args:
    - dataset (str): The path to the CSV dataset containing user IDs.

    Returns:
    - list: A list of user IDs loaded from the dataset.
    """

    user_matrix = pd.read_csv(dataset)
    return user_matrix["userId"].tolist()

# List from load_user_ids saved in variable users
#users = load_user_ids("src/data/processed/user_matrix.csv")    # pour les tests API

# Use SQL Alchemy engine
engine, inspector = get_engine()

# Generate a list of movies unrated by the user
query = "SELECT * FROM users;"
users = pd.read_sql(query, engine)
users = users['user_id'].to_list()



def sign_jwt(user_id: int):
    """
    Description:
    Generate a JWT token that expires after 10min with a user ID payload.

    Args:
    - user_id (int): The user ID to include in the JWT payload.

    Returns:
    - str: The generated JWT token.
    """

    payload = {"user_id": user_id, "expires": time.time() + 600}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def decode_jwt(token: str):
    """
    Description:
    Decode a JWT token and verify its validity.

    Args:
    - token (str): The JWT token to decode and verify.

    Returns:
    - dict or None: The decoded payload if the token is valid and not expired, otherwise None.
    """

    try:
        decoded_token = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return decoded_token if decoded_token["expires"] >= time.time() else None
    except jwt.JWTError:
        return None

class JWTBearer(HTTPBearer):
    """
    Description:
    Class for JWT token authentication.

    Methods:
    - __init__: Initialize the JWTBearer object.
    - __call__: Validate the JWT token from the request.
    - verify_jwt: Verify the validity of the JWT token.
    """

    def __init__(self, auto_error: bool = True):
        """
        Description:
        Initialize the JWTBearer object.

        Args:
        - auto_error (bool): Determines whether to raise an HTTPException automatically if the token
          is invalid or expired. Default is True.
        """

        super(JWTBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        """
        Description:
        Validate the JWT token from the request.

        Args:
        - request (Request): The incoming request object.

        Returns:
        - int or False: The user ID extracted from the JWT token if it's valid, otherwise False.
          If the token is invalid or expired, or if the request does not contain a token, it raises an HTTPException.
        """

        # Extract the credentials (JWT token) from the request using the HTTPBearer authentication
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)
        if credentials:
            # Check if the token scheme is "Bearer"
            if not credentials.scheme == "Bearer":
                raise HTTPException(
                    status_code=403, detail="Invalid authentication scheme."
                )
            
            # Verify the JWT token and extract the user ID
            user_id = self.verify_jwt(credentials.credentials)
            if not user_id:
                raise HTTPException(
                    status_code=403, detail="Invalid token or expired token."
                )
            return user_id
        
        # Raise an HTTPException with status code 403 and detail message if no token is provided in the request
        else:
            raise HTTPException(
                status_code=403, detail="Invalid authorization code."
            )

    def verify_jwt(self, jwtoken: str):
        """
        Description:
        Verify the validity of a JWT token.

        Args:
        - jwtoken (str): The JWT token to verify.

        Returns:
        - int or False: The user ID if the token is valid and belongs to an authorized user, otherwise False.
        """
        
        try:
            payload = decode_jwt(jwtoken)
            if payload is None:
                return False
            user_id = payload.get("user_id")
            if user_id is None or user_id not in users:
                return False
            return user_id
        except Exception:
            return False
    
        