import redis
import os
import base64
from binascii import hexlify

from threading import Thread, BoundedSemaphore
from waitress import serve
from flask import Flask, Blueprint, request, Response, abort, jsonify, send_file, make_response, send_from_directory
from wtforms import Form, IntegerField, StringField, BooleanField
from wtforms.validators import DataRequired, ValidationError

from spotify_helper import find_track
from machine_learning import ska_prob, load_clfs, ClfInfo, transform, avg_value, create_fresh_clf
from track_update import track_updater_worker, add_track_update
from env import STATIC_FILE_PATH, REDIS_IP, REDIS_PORT, REDIS_ACCESS_DB, MASTER_ACCESS_TOKEN

app = Flask(__name__, static_folder=STATIC_FILE_PATH)

def validate_track_name(form, field):
	if(len(field.data) == 0):
		raise ValidationError('invalid name')

class SkaProbForm(Form):
	track_name = StringField(
		"track_name",
		[
			DataRequired(),
			validate_track_name
		]
	)

class SkaCorrectionForm(Form):
	ska = BooleanField(
		"ska",
		[
		]
	)
	track_id = StringField(
		"track_id",
		[
			DataRequired()
		]
	)
	# access_token = StringField(
	# 	"access_token",
	# 	[
	# 		DataRequired()
	# 	]
	# )

class CreateAccessToken(Form):
	master_key = StringField(
		"master_key",
		[
			DataRequired()
		]
	)

_redis_access_sem = BoundedSemaphore(1)
_reds_conn = redis.Redis(
		host=REDIS_IP,
		port=REDIS_PORT,
		db=REDIS_ACCESS_DB
	)

ACCESS_TOKEN_KEY = "access_tokens"

def create_access_token():
	_redis_access_sem.acquire()	
	try:
		access_token = base64.b64encode(os.urandom(20)).decode("utf-8")
		_reds_conn.rpush(
			ACCESS_TOKEN_KEY,
			access_token
		)
	finally:
		_redis_access_sem.release()

	return access_token

def check_access_token(access_token):
	result = False
	_redis_access_sem.acquire()
	try:
		for i in range(_reds_conn.llen(ACCESS_TOKEN_KEY)):
			if access_token == _reds_conn.lindex(ACCESS_TOKEN_KEY, i):
				result = True
				break
	finally:
		_redis_access_sem.release()

	return result

@app.route("/api/ska_prob", methods=["GET"])
def ska_prob_endpoint():
	form = SkaProbForm(request.args)

	if not form.validate():
		return make_response(
			jsonify({
				"error" : form.errors
			}),
			400
		)
	
	track_info = find_track(form.track_name.data)
	if track_info == None:
		return make_response(
			jsonify({
				"error" : "track could not be found"
			}),
			404
		)

	prob = ska_prob(track_info["id"])

	album_image_url = None
	if len(track_info["album"]["images"]) > 0:
		album_image_url = track_info["album"]["images"][0]["url"]

	return {
		"track_id" : track_info["id"],
		"prob" : prob,
		"album" : track_info["album"]["name"],
		"title" : track_info["name"],
		"artists" : [i["name"] for i in track_info["artists"]],
		"album_image_url" : album_image_url,
		"track_link" : "https://open.spotify.com/track/{}".format(track_info["id"])
	}

@app.route("/api/correction", methods=["POST"])
def ska_correction_endpoint():
	form = SkaCorrectionForm(request.form)

	if not form.validate():
		return make_response(
			jsonify({
				"error" : form.errors
			}),
			400
		)

	# access_token = form.access_token.data.encode("utf-8")
	# if not check_access_token(access_token):
	# 	return make_response(
	# 		jsonify({
	# 			"error" : "invalid access token"
	# 		}),
	# 		403
	# 	)

	add_track_update(
		track_id=form.track_id.data,
		is_ska=form.ska.data,
		ip_address=request.remote_addr
	)
	
	return make_response(
		jsonify({
			"result" : "success"
		}),
		200
	)


@app.route("/api/create_access_token", methods=["POST"])
def create_access_token_endpoint():
	form = CreateAccessToken(request.form)

	if not form.validate():
		return make_response(
			jsonify({
				"error" : form.errors
			}),
			400
		)

	if form.master_key.data != MASTER_ACCESS_TOKEN:
		return make_response(
			jsonify({
				"error" : "invalid access token"
			}),
			403
		)

	return make_response(
		jsonify({
			"result" : create_access_token()
		}),
		200
	)


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def default_path_endpoint(path):
	if path != "" and os.path.exists(os.path.join(app.static_folder, path).strip()):
		return send_from_directory(app.static_folder, path)
	else:
		return send_from_directory(app.static_folder, 'index.html')

def main():
	# create_fresh_clf()
	# return
	load_clfs()
	Thread(target=track_updater_worker).start()
	serve(app, host="0.0.0.0", port=7000)

if __name__ == "__main__":
	main()