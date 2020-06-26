import redis
import json

from queue import Queue

from spotify_helper import get_playlist_tracks, add_track_to_playlist, get_playlist_by_name, remove_track_from_playlist
from env import SKA_PLAYLIST, NOT_SKA_PLAYLIST, TRACK_VOTING_DB, REDIS_IP, REDIS_PORT

_workQueue = Queue()

_ska_playlist_id, _ska_playlist_n = get_playlist_by_name(
	name=SKA_PLAYLIST
)

_ska_tracks = get_playlist_tracks(
	play_name=SKA_PLAYLIST
)

_not_ska_playlist_id, _not_ska_playlist_n = get_playlist_by_name(
	name=NOT_SKA_PLAYLIST
)

_not_ska_tracks = get_playlist_tracks(
	play_name=NOT_SKA_PLAYLIST
)

class WorkTask():
	def __init__(self, track_id, is_ska, ip_address):
		super().__init__()
		self.track_id = track_id
		self.is_ska = is_ska
		self.ip_address = ip_address

def get_track_add_key(track_id):
	return "track:vote:{}".format(track_id)

def process(rconn, top):
	track_key = get_track_add_key(top.track_id)

	voting_count = rconn.llen(
		track_key
	)

	votes = []
	voting_sum = 0
	for i in range(0, voting_count):
		vote = json.loads(rconn.lindex(track_key, i))
		voting_sum += vote["ska"]
		if "ip_address" in vote and vote["ip_address"] == top.ip_address:
			return
	
	# Is ska
	if voting_sum > 2:
		if top.track_id in _not_ska_tracks:
			remove_track_from_playlist(_not_ska_playlist_id, top.track_id)

		if not top.track_id in _ska_tracks:
			add_track_to_playlist(_ska_playlist_id, top.track_id)
			_ska_tracks.append(top.track_id)
	# Is not ska
	elif voting_sum < -2:
		if top.track_id in _ska_tracks:
			remove_track_from_playlist(_ska_tracks, top.track_id)

		if not top.track_id in _not_ska_tracks:
			add_track_to_playlist(_not_ska_playlist_id, top.track_id)
			_not_ska_tracks.append(top.track_id)

	rconn.rpush(
		track_key,
		json.dumps({
			"ska": 1 if top.is_ska else -1,
			"ip_address": top.ip_address
		})
	)

def track_updater_worker():
	rconn = redis.Redis(
		host=REDIS_IP,
		port=REDIS_PORT,
		db=TRACK_VOTING_DB
	)

	while True:
		top = _workQueue.get()
		try:
			process(rconn, top)

		except Exception as e:
			print("failure voting on song " + e)

def add_track_update(track_id, is_ska, ip_address):
	_workQueue.put(
		WorkTask(
			track_id=track_id,
			is_ska=is_ska,
			ip_address=ip_address
		)
	)