import struct
import time
import copy
import os
import multiprocessing
import graphviz
import pickle
import io
import itertools
import hashlib
import re
import random
import redis
import json
import gzip
import spotipy

import pandas as pd
import numpy as np

from io import StringIO  # Python 3.x
# Import train_test_split function
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics, tree, preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from datetime import datetime
from multiprocessing import Process, JoinableQueue
from threading import Semaphore, Thread

from env import SOURCE_PLAYLIST, SKA_PLAYLIST, CLF_FOLDER_PATH
from spotify_helper import get_playlist_tracks, get_features_for_tracks, get_ska_playlist


FEATURES = [
    "beats"
]

_loaded_clf = None


class ClfInfo():
    clf = None
    feature_set = None


def get_feature_key(fun, key, other_key, n, id):
    return "{}:{}:{}:{}:{}".format(fun, key, other_key, n, id)


def compress_string(data):
    return gzip.compress(data.encode("utf-8"))


def decompress_to_string(data):
    return gzip.decompress(data).decode("utf-8")


def transform(complete_track, key, other_key, n, redis_connection=None):
    feature_key = get_feature_key(
        transform.__name__, key, other_key, n, complete_track["id"])

    result = {}
    for i in range(0, n):
        if i < len(complete_track["analysis"][key]):
            val = float(complete_track["analysis"][key][i][other_key])
        else:
            val = float(-1)

        result["{}{}{}".format(i, key, other_key)] = val

    return result


def avg_value(complete_track, key, other_key, n, redis_connection=None):
    feature_key = get_feature_key(
        avg_value.__name__, key, other_key, n, complete_track["id"])

    result = {}

    for i in range(0, n):
        if i < len(complete_track["analysis"][key]):
            val = sum(j for j in complete_track["analysis"][key][i][other_key]) / len(
                complete_track["analysis"][key][i][other_key])
        else:
            val = float(-1)

        result["{}{}{}".format(i, key, other_key)] = val

    return result


def string_to_bytes(complete_track, key, other_key, n, redis_connection=None):

    feature_key = get_feature_key(
        string_to_bytes.__name__, key,
        other_key, n, complete_track["id"])

    result = {}

    for i in range(0, n):
        if i < len(complete_track["analysis"][key][other_key]):
            x = complete_track["analysis"][key][other_key][i]
            val = float(ord(x))
        else:
            val = -1

        result["{}{}{}".format(i, key, other_key)] = val

    return result


def get_ml_features(
    complete_track,
    feature_set,
    ska_tracks=None,
):
    result = {}

    for key in feature_set:
        result[key] = int(complete_track['features'][key] * 1000)

    if ska_tracks != None:
        if complete_track["id"] in ska_tracks:
            result["ska"] = float(True)
        else:
            result["ska"] = float(False)

    return result


def gen_descion_tree_clf():
    return tree.DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=3,
        max_features="log2"
    )


def gen_random_forest_clf():
    return RandomForestClassifier(
        max_depth=None,
        n_estimators=1000,
        min_samples_leaf=3,
        random_state=None
    )


def gen_nearest_neighbors_clf():
    return NearestNeighbors(
        n_neighbors=5,
        algorithm='ball_tree'
    )


def gen_neural_network_clf():
    return MLPClassifier(
        solver='adam',
        hidden_layer_sizes=(100, 50,),
        random_state=1,
        max_iter=50000
    )


def gen_classifier(clf_gen, target_var, df, features):
    x = df[features]  # Features
    y = df[target_var]  # Target variable

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=1)  # 70% training and 30% test

    clf = clf_gen()
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    return clf, accuracy_score(y_test, y_pred)


def _is_it_ska(
        feature_set,
        track_id,
        clf,
):
    track_complete = {
        "id": track_id,
        "features": get_features_for_tracks([{'id': track_id}])[0]
    }

    ml_set = get_ml_features(
        complete_track=track_complete,
        feature_set=feature_set,
    )

    cols = list(ml_set.keys())
    df = pd.DataFrame([ml_set], columns=cols)

    x = df[cols]

    return clf.predict_proba(x)


def ska_prob(track_id):
    clfInfo = _loaded_clf

    return _is_it_ska(
        clf=clfInfo.clf,
        feature_set=clfInfo.feature_set,
        track_id=track_id
    )[0][1]


def get_track_data():
    training_tracks = get_playlist_tracks(
        playlist_id=SOURCE_PLAYLIST
    )

    ska_tracks = get_ska_playlist()

    track_features = get_features_for_tracks(
        tracks=training_tracks
    )

    def processTrainingTracks(i):
        return {
            "id": training_tracks[i],
            "features": track_features[i]
        }

    tracks_complete = []
    for i in range(0, len(training_tracks)):
        tracks_complete.append(
            processTrainingTracks(i)
        )

    return tracks_complete, ska_tracks


def get_ml_pipeline(
        complete_track,
        feature_set,
        length_dict,
        ska_tracks,
        pipeline,
        redis_connection
):
    for i in feature_set:
        key = i["key"]
        sec = i["sec"]
        if key in length_dict:
            n = length_dict[key]
        elif sec in length_dict:
            n = length_dict[sec]
        else:
            raise Exception("missing entry in legnth dict")

        feature_key = get_feature_key(
            i["fun"].__name__,
            key,
            sec,
            n,
            complete_track["id"]
        )

        pipeline.get(feature_key)


def create_clf(
    tracks_complete,
    ska_tracks,
    feature_set,
):
    ml_set = [get_ml_features(
        complete_track=i,
        feature_set=feature_set,
        ska_tracks=ska_tracks,
    ) for i in tracks_complete]

    cols = list(ml_set[0].keys())
    df = pd.DataFrame(ml_set, columns=cols)
    cols.remove("ska")

    clf, acc = gen_classifier(
        clf_gen=gen_random_forest_clf,
        target_var="ska",
        features=cols,
        df=df,
    )

    toDump = ClfInfo()
    toDump.clf = clf
    toDump.feature_set = feature_set

    dump = pickle.dumps(toDump)

    return dump, "clf.bin", acc


def get_top_x_clf(x, percent=False):
    ratings = []

    target_floder = CLF_FOLDER_PATH

    for filename in os.listdir(target_floder):
        ratings.append({
            "rating": int(filename[:3]),
            "filename": "{}/{}".format(target_floder, filename)
        })

    ratings.sort(key=lambda x: x["rating"], reverse=True)

    if percent:
        length = int(len(ratings) * x)
    else:
        length = x
    return ratings[:min(length, len(ratings))]


def load_clf():

    full_path = os.path.join(CLF_FOLDER_PATH, "clf.bin")

    if not os.path.isfile(full_path):
        return None

    clfInfo = pickle.load(open(full_path, 'rb'))

    return clfInfo


def create_fresh_clf():
    tracks_complete, ska_tracks = get_track_data()

    tracks_complete_dump = json.dumps(tracks_complete)
    ska_tracks_dump = json.dumps(ska_tracks)

    delta_histroy = []
    feature_set = [
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms"
    ]

    dump, name, acc = create_clf(
        ska_tracks=ska_tracks,
        tracks_complete=tracks_complete,
        feature_set=feature_set,
    )

    print("created new CLF with acc:{}".format(acc))

    with open(os.path.join(CLF_FOLDER_PATH, name), "wb") as f:
        f.write(dump)

    return load_clf()


def init_clf():
    global _loaded_clf

    clf = load_clf()
    if clf == None:
        clf = create_fresh_clf()

    _loaded_clf = clf
