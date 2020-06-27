import sys
import random
import requests
import json
import editdistance
import random
import time
import redis
import multiprocessing

import numpy as np
import urllib.parse

from datetime import datetime, timedelta
from os import path
from threading import BoundedSemaphore
from joblib import Parallel, delayed

from env import API_AUTH, REFRESH_TOKEN, REDIS_IP, REDIS_PORT, REDIS_DB

DEFAULT_BACKOFF = 1
POST = "POST"
PUT = "PUT"
GET = "GET"
DELETE = "DELETE"

_r = {}
_api_auth = None
_refresh_token = None

_access_token = None
_access_token_expire_time = None

_request_sem = BoundedSemaphore(3)

def get_time_key(key):
	return "{}_time".format(key)

def get_cache_time():
	return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")

def get_from_cache(key, expire_time_secs=None):
	if expire_time_secs != None:
		if not _r.exists(get_time_key(key)):
			_r.set(get_time_key(key), get_cache_time())

		inserted_time = datetime.datetime.strptime(
			_r.get(get_time_key(key)),
			"%Y-%m-%d %H:%M:%S.%f"
		)
		delta = (datetime.utcnow() - inserted_time).total_seconds()
		if delta > expire_time_secs:
			return None

	if _r.exists(key):
		return json.loads(_r.get(key))

	return None

def update_cache(key, value):
	_r.set(key, json.dumps(value))
	_r.set(get_time_key(key), get_cache_time())

def init_cache():
	global _r

	_r = redis.Redis(
		host=REDIS_IP,
		port=REDIS_PORT,
		db=REDIS_DB
	)

def set_login(api_auth=None, refresh_token=None):
	global _api_auth
	global _refresh_token

	_api_auth = api_auth
	_refresh_token = refresh_token

def get_access_token():
	global _access_token_expire_time
	global _access_token

	if _api_auth == None or _refresh_token == None:
		raise Exception("Must set login first!")

	if _access_token == None or datetime.utcnow() > _access_token_expire_time:
		_access_token, _access_token_expire_time = get_access_token_request(
			api_auth=_api_auth,
			refresh_token=_refresh_token
		)
	
	return _access_token

def get_header(access_token=None):
	return {
		'Content-Type': "application/json",
		'Authorization': "Bearer {}".format(
			get_access_token()
		),
		'Connection': "keep-alive",
		'cache-control': "no-cache"
	}

def make_request(
	url,
	method="GET",
	headers=None,
	payload="",
	backoff=DEFAULT_BACKOFF,
	params={},
	calls=0
):
	if(headers == None):
		headers = get_header()

	if(calls > 7):
		print("Too many backoffs")
		return None

	_request_sem.acquire()
	try:
		response = requests.request(
			method,
			url,
			data=payload,
			headers=headers,
			timeout=None,
			params=params
		)
	finally:
		_request_sem.release()

	if response.status_code > 500 and response.status_code < 599:
		print(
			"Code {} trying again in {} Error {}".format(
				response.status_code,
				backoff,
				response.text
			)
		)
		time.sleep(backoff)
		return make_request(
			url=url,
			method=method,
			headers=headers,
			payload=payload,
			backoff=backoff * 2,
			calls=calls + 1
		)
	
	if response.status_code >= 400 and response.status_code <= 499:
		print("Error code on {} URL: {} Error: {}".format(response.status_code, url, response.text))
		return None

	return response

def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def get_or_fetch(url, key, expire_time_secs=None):
	cache_result = get_from_cache(
		key=key
	)

	if cache_result != None:
		return cache_result

	# time.sleep(random.randint(0, 100))
	response = make_request(url)
	
	if(response == None):
		print("failled to get")
		return None

	result = json.loads(response.text)

	update_cache(
		key=key,
		value=result
	)

	return result


def get_access_token_request(api_auth, refresh_token):
	url = "https://accounts.spotify.com/api/token?grant_type=refresh_token&refresh_token={}".format(refresh_token)

	headers = {
		'Authorization': 'Basic %s' % api_auth, 
		'Content-Type': 'application/x-www-form-urlencoded'
	}
	response = make_request(url, method=POST, headers=headers)

	response_json = json.loads(response.text)

	if "expires_in" not in response_json:
		print("api_auth {} refresh_token {}".format(api_auth, refresh_token))
		print("Error getting Spotify access token only found " + json.dumps(response_json))

	expire_delta = timedelta(seconds=response_json["expires_in"] - 10)
	expire_time = datetime.utcnow() +  expire_delta
	return response_json["access_token"], expire_time

def get_track_features(track_id=None):
	url = "https://api.spotify.com/v1/audio-features/" + track_id
	key = "feature:{}".format(track_id)
	return get_or_fetch(key=key, url=url)

def get_playlist_tracks(play_name=None, play_id=None, n=None):

	if play_name != None:
		play_id, n = get_playlist_by_name(name=play_name)
	elif play_id == None or n == None:
		raise Exception("Must set play_name or (play_id and n)")

	offset = 0

	result = []

	while(offset < n):
		url = "https://api.spotify.com/v1/playlists/{}/tracks?tracks=100&fields=items(track(id))&offset={}".format(play_id, offset)
		response = make_request(url)
		offset += 100

		for i in json.loads(response.text)["items"]:
			result.append(i)

	print("Playlist {} gotten".format(play_id))
	return [i["track"]["id"] for i in result]

def get_playlist_by_name(name):
	print("Getting playlist named {}".format(name))

	url = "https://api.spotify.com/v1/me/playlists"
	response = make_request(url, GET)

	if response == None:
		return None, None

	for i in json.loads(response.text)["items"]:
		x = i["name"].lower()
		if editdistance.distance(name.lower(), x.lower()) < 3:
			return i["id"], i["tracks"]["total"]

	return None, None

def add_song_to_queue(track_id=None):
	track_uri = "spotify:track:{}".format(track_id)
	url = "https://api.spotify.com/v1/me/player/queue?uri=%s" % track_uri
	response = make_request(url, method=POST)

def play_song():
	url = "https://api.spotify.com/v1/me/player/play"
	make_request(url, method=PUT)

def pause_song():
	url = "https://api.spotify.com/v1/me/player/pause"
	make_request(url, method=PUT)

def skip_song():
	url = "https://api.spotify.com/v1/me/player/next"
	response = make_request(url, method=POST)

def currently_playing(retry_until_playing=True):
	url = "https://api.spotify.com/v1/me/player/currently-playing"
	response = make_request(url)

	if response.status_code == 204 and retry_until_playing:
		print("failled to skip retrying")
		time.sleep(5 + random.randint(0, 10) + random.random())
		return currently_playing()

	return json.loads(response.text)

def skip_until_song(track_id=None, max_tries=2):
	def playing_id():
		return currently_playing()["item"]["id"]

	i = 0
	while(playing_id() != track_id and i < max_tries):
		skip_song()
		time.sleep(0.5)
		i += 1

def get_features_for_tracks(tracks=[]):
	result = []

	for i in range(0, len(tracks)):
		print_progress_bar(i, len(tracks), prefix="Getting track features")
		result.append(get_track_features(tracks[i]))

	print_progress_bar(i, len(tracks), prefix="Getting track features")
	return result

def get_analysis_for_track(track_id=None):
	key = "analysis:{}".format(track_id)
	url = "https://api.spotify.com/v1/audio-analysis/{}".format(track_id)
	return get_or_fetch(url, key)

def get_analysis_for_tracks(tracks=None):
	result = []

	for i in range(0, len(tracks)):
		result.append(
			get_analysis_for_track(tracks[i])
		)
		print_progress_bar(i, len(tracks))

	print_progress_bar(len(tracks), len(tracks))

	return result

def get_track(track_id=""):
	key = "track:{}".format(track_id)
	url = "https://api.spotify.com/v1/tracks/{}".format(track_id)
	return get_or_fetch(url, key)

def get_album(album_id=""):
	key = "album:{}".format(album_id)
	url = "https://api.spotify.com/v1/albums/{}".format(album_id)
	return get_or_fetch(url, key)

def get_artist(artist_id=""):
	key = "artist:{}".format(artist_id)
	url = "https://api.spotify.com/v1/artists/{}".format(artist_id)
	return get_or_fetch(url, key)

def get_genres_for_track(track_id=""):
	track_info = get_track(track_id)
	album = get_album(track_info["album"]["id"])
	if len(album["genres"]) > 0:
		return album["genres"]
	
	genres = []
	for i in album["artists"]:
		artist = get_artist(artist_id=i["id"])
		genres.extend(
			artist["genres"]
		)

	return genres

def add_tracks_to_playlist(playlist_id, tracks):
	url = "https://api.spotify.com/v1/playlists/{}/tracks".format(playlist_id)

	for i in range(0, len(tracks), 100):
		payload = {
			"uris" : ["spotify:track:{}".format(j) for j in tracks[i:min(i + 100, len(tracks))]]
		}
		response = make_request(url, method=POST, payload=json.dumps(payload))

def add_track_to_playlist(playlist_id, track_id):
	return add_tracks_to_playlist(
		playlist_id=playlist_id,
		tracks=[track_id]
	)

def search(search_query, search_type="track"):
	url = "https://api.spotify.com/v1/search"
	parmas = {
		"q" : search_query,
		"type" : search_type
	}

	parsed_url = "{}?{}".format(
		url,
		urllib.parse.urlencode(parmas)
	)

	return get_or_fetch(parsed_url, key=parsed_url)

def find_closest(ary, target, get_fun, dist_fun):
	min_dist = sys.maxsize
	best_idx = 0

	for i in range(0, len(ary)):
		dist = dist_fun(target, get_fun(ary, i))
		if dist < min_dist:
			min_dist = dist
			best_idx = i

	return best_idx

def find_track(track_name, artist_name=None):
	response = search(track_name, search_type="track")

	def get_track_name(ary, i):
		return ary[i]

	def dist_fun(target, other):
		result = editdistance.distance(
			target.lower(),
			other["name"].lower()
		)
		
		if artist_name != None:
			result += min(editdistance.distance(i["name"].lower(), artist_name.lower()) for i in other["artists"])

		return result
				
	if len(response["tracks"]["items"]) == 0:
		return None

	idx = find_closest(
		response["tracks"]["items"],
		track_name,
		get_track_name,
		dist_fun
	)

	return response["tracks"]["items"][idx]

def remove_tracks_from_playlist(playlist_id, tracks):
	url = "https://api.spotify.com/v1/playlists/{}/tracks".format(
		urllib.parse.quote(playlist_id)
	)

	for i in range(0, len(tracks), 100):
		selected = tracks[i:min(i + 100, len(tracks))]
		body = {
			"tracks" : [{"uri" : "spotify:track:{}".format(j) } for j in selected]
		}
		response = make_request(
			url,
			method=DELETE,
			payload=json.dumps(body)
		)

def remove_track_from_playlist(playlist_id, track_id):
	return remove_tracks_from_playlist(
		playlist_id=playlist_id,
		tracks=[track_id]
	)

init_cache()

set_login(
	api_auth=API_AUTH,
	refresh_token=REFRESH_TOKEN,
)
