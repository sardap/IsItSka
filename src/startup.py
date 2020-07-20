import re
import redis
import os
import base64

from threading import Thread, BoundedSemaphore
from flask import Flask, Blueprint, request, Response, abort, jsonify, send_file, make_response, send_from_directory
from wtforms import Form, IntegerField, StringField, BooleanField
from wtforms.validators import DataRequired, ValidationError, Optional

from spotify_helper import find_track, get_track, spotify_helper_init
from machine_learning import ska_prob, load_clfs, ClfInfo, transform, avg_value, create_fresh_clf, clf_refresher_worker, get_feature_score
from track_update import track_updater_worker, add_track_update, init_track_update
from env import STATIC_FILE_PATH, REDIS_IP, REDIS_PORT, REDIS_ACCESS_DB, MASTER_ACCESS_TOKEN, PORT

app = Flask(__name__, static_folder=STATIC_FILE_PATH)

def validate_track_name(form, field):
	if(len(field.data) == 0):
		raise ValidationError('invalid name')

class SkaProbForm(Form):
	track_name = StringField(
		"track_name",
		[
			Optional(),
			validate_track_name
		]
	)
	artist_name = StringField(
		"artist_name",
		[
		]
	)
	track_id = StringField(
		"track_id",
		[
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

class CreateAccessToken(Form):
	master_key = StringField(
		"master_key",
		[
			DataRequired()
		]
	)

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

	if len(form.track_id.data) == 0 and len(form.track_name.data) == 0:
		return make_response(
			jsonify({
				"error" : "must set track id or track name"
			}),
			400
		)

	artist_name = form.artist_name.data
	if len(artist_name) == 0:
		artist_name = None

	if len(form.track_name.data) > 0:
		track_info = find_track(
			form.track_name.data,
			artist_name=artist_name
		)
	else:
		track_id = form.track_id.data
		if not re.match("^[a-zA-Z0-9_]+$", track_id):
			track_id = track_id.split("/")[4]
			if len(track_id.split("?")) > 0:
				track_id = track_id.split("?")[0]

		track_info = get_track(track_id)

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

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def default_path_endpoint(path):
	if path != "" and os.path.exists(os.path.join(app.static_folder, path).strip()):
		return send_from_directory(app.static_folder, path)
	else:
		return send_from_directory(app.static_folder, 'index.html')

def main():
	print("starting server")
	spotify_helper_init()
	print("Loading classfiers")
	load_clfs()
	print("initalsing track updater")
	init_track_update()
	print("starting Track workers")
	Thread(target=track_updater_worker).start()
	print("staring clf refresher")
	Thread(target=clf_refresher_worker).start()

	print("starting webserver")
	app.run(host="0.0.0.0", port=PORT, threaded=True)
	# serve(app, host="0.0.0.0", port=PORT)

if __name__ == "__main__":
	main()