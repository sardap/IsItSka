import base64
import os

SKA_PLAYLIST = os.environ["SKA_PLAYLIST"]
NOT_SKA_PLAYLIST = os.environ["NOT_SKA_PLAYLIST"]
SOURCE_PLAYLIST = os.environ["SOURCE_PLAYLIST"]

API_AUTH = os.environ["API_AUTH"]

def get_refresh_token():
	tmp = os.environ["REFRESH_TOKEN"]
	tmp_encoded = tmp.encode("utf-8")
	tmp_bytes = base64.b64decode(tmp_encoded)
	return tmp_bytes.decode("utf-8")

REFRESH_TOKEN = get_refresh_token()

CLF_FOLDER_PATH = os.environ["CLF_FOLDER_PATH"]
STATIC_FILE_PATH = os.environ["STATIC_FILE_PATH"]

REDIS_IP = os.environ["REDIS_IP"]
REDIS_PORT = os.environ["REDIS_PORT"]
REDIS_DB = os.environ["REDIS_DB"]
REDIS_ACCESS_DB = os.environ["REDIS_ACCESS_DB"]
REDIS_ML_DB = os.environ.get("REDIS_ML_DB", "3")

MASTER_ACCESS_TOKEN = os.environ["MASTER_ACCESS_TOKEN"]

TRACK_VOTING_DB = os.environ["TRACK_VOTING_DB"]

PORT = os.environ["PORT"]