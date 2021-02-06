import re
import redis
import os
import base64
import random

from threading import Thread, BoundedSemaphore
from flask import Flask, Blueprint, request, Response, abort, jsonify, send_file, make_response, send_from_directory
from wtforms import Form, IntegerField, StringField, BooleanField
from wtforms.validators import DataRequired, ValidationError, Optional, NumberRange
from waitress import serve

from spotify_helper import get_spotipy_client, find_track, get_playlist_tracks, get_ska_playlist
from machine_learning import ska_prob, init_clf
from env import STATIC_FILE_PATH, PORT, SKA_PLAYLIST

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


class SomeSka(Form):
    n = IntegerField(
        "n",
        [
            DataRequired(),
            NumberRange(min=0, max=10)
        ]
    )


def to_track_out(track_info, prob=-1):
    album_image_url = None
    if len(track_info["album"]["images"]) > 0:
        album_image_url = track_info["album"]["images"][0]["url"]

    return {
        "track_id": track_info["id"],
        "prob": prob,
        "album": track_info["album"]["name"],
        "title": track_info["name"],
        "artists": [i["name"] for i in track_info["artists"]],
        "album_image_url": album_image_url,
        "track_link": "https://open.spotify.com/track/{}".format(track_info["id"])
    }


@app.route("/api/ska_prob", methods=["GET"])
def ska_prob_endpoint():
    form = SkaProbForm(request.args)

    if not form.validate():
        return make_response(
            jsonify({
                "error": form.errors
            }),
            400
        )

    if len(form.track_id.data) == 0 and len(form.track_name.data) == 0:
        return make_response(
            jsonify({
                "error": "must set track id or track name"
            }),
            400
        )

    artist_name = form.artist_name.data
    if len(artist_name) == 0:
        artist_name = None

    if len(form.track_name.data) > 0:
        track_info = find_track(
            track_name=form.track_name.data,
            artist_name=artist_name)

    else:
        track_id = form.track_id.data
        if not re.match("^[a-zA-Z0-9_]+$", track_id):
            track_id = track_id.split("/")[4]
            if len(track_id.split("?")) > 0:
                track_id = track_id.split("?")[0]

        track_info = get_spotipy_client().track(track_id)

    if track_info == None:
        return make_response(
            jsonify({
                "error": "track could not be found"
            }),
            404
        )

    prob = ska_prob(track_info["id"])

    return to_track_out(track_info, prob)


@app.route("/api/some_ska", methods=["GET"])
def some_ska_endpoint():
    form = SomeSka(request.args)

    if not form.validate():
        return make_response(
            jsonify({
                "error": form.errors
            }),
            400
        )

    ska_tracks = get_ska_playlist()

    random.shuffle(ska_tracks)

    tracks = []
    for track in ska_tracks[:form.n.data]:
        tracks.append(to_track_out(get_spotipy_client().track(track['id'])))

    return make_response(
        jsonify({
            "tracks": tracks
        }),
        200
    )


@app.route("/api/correction", methods=["POST"])
def ska_correction_endpoint():
    return {}, 500


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def default_path_endpoint(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path).strip()):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


init_clf()

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=PORT)
